"""Bounded, fully checkpointed SHINE learnability diagnostics (not stage A)."""

from dataclasses import asdict
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dsrsi.data.compiler import EpisodeCompiler, tokenizer_fingerprint
from dsrsi.data.shine import SHINE_DATA_VERSION, shine_episode
from dsrsi.evaluation.fixed_records import evidence_teacher_response
from dsrsi.evaluation.m_only import normalized_answer
from dsrsi.evaluation.journal import EvaluationJournal, EvaluationPaused, digest
from dsrsi.model.shine import GeneratedMemory, ShineConfig, ShineQwen
from dsrsi.utils.atomic import atomic_json
from .checkpoint import (CheckpointStore, Progress, capture_training_state, restore_training_state,
                         capture_rng, restore_rng, file_sha256)
from .control import ControlChannel
from .device import resolve_device
from .diagnostic_budget import DiagnosticBudget, DiagnosticBudgetExceeded


class DirectFactors(nn.Module):
    """Slow per-context fitted adapters, a capacity diagnostic only."""
    def __init__(self, model, rows):
        super().__init__()
        self.bank = nn.ParameterDict()
        self.config = model.config
        self.input_width, self.output_width = model.input_width, model.output_width
        for number, row in enumerate(rows):
            for part in range(len(row["source_events"])):
                for layer in self.config.layers:
                    key = f"c{number}_p{part}_l{layer}"
                    a = torch.empty(self.config.rank, self.input_width)
                    nn.init.kaiming_uniform_(a, a=5 ** .5)
                    self.bank[key + "a"] = nn.Parameter(a)
                    self.bank[key + "b"] = nn.Parameter(torch.zeros(self.output_width, self.config.rank))

    def memories(self, number, row):
        return [GeneratedMemory(source["event_id"], row["episode_id"],
                    {layer: (self.bank[f"c{number}_p{part}_l{layer}a"],
                             self.bank[f"c{number}_p{part}_l{layer}b"]) for layer in self.config.layers})
                for part, source in enumerate(row["source_events"])]


def fixture_rows(*, worlds, task, seed, language="zh", split="train"):
    return [shine_episode(w, variant=v, task=task, seed=seed, language=language, split=split)
            for w in range(worlds) for v in (0, 1)]


def encode_memories(model, episode):
    device = model.base.embed_tokens.weight.device
    memories = [model.encode(torch.tensor([source.input_ids], device=device),
                             record_id=source.event_id, life_id=episode.episode_id) for source in episode.source]
    cost = sum(len(s.input_ids) + model.memory_tokens for s in episode.source)
    return memories, cost


def objective(model, compiler, row, *, direct=None, context_index=None, counterfactual=None, cf_weight=1.0):
    episode = compiler.compile(row)
    device = model.base.embed_tokens.weight.device
    if direct is None:
        memories, total = encode_memories(model, episode)
    else:
        memories, total = direct.memories(context_index, row), 0
    other_memories = other_episode = None
    if cf_weight:
        if counterfactual is None or counterfactual["world_id"] != row["world_id"]:
            raise ValueError("SHINE counterfactual world differs")
        other_episode = compiler.compile(counterfactual)
        if direct is None:
            other_memories, extra = encode_memories(model, other_episode)
            total += extra
        else:
            other_memories = direct.memories(context_index ^ 1, counterfactual)
    answers, contrasts, losses = [], [], []
    for number, response in enumerate(episode.responses):
        ids = torch.tensor([response.input_ids], device=device)
        _, value = model(ids, memories=memories, life_id=episode.episode_id,
                         labels=torch.tensor([response.labels], device=device))
        answer = value.mean()
        total += len(response.input_ids)
        contrast = answer.detach().new_zeros(())
        if cf_weight:
            opposite = other_episode.responses[number]
            if response.prompt_ids != opposite.prompt_ids:
                raise ValueError("SHINE content variants changed the question")
            targets = response.input_ids[response.prompt_length:]
            alternate = opposite.input_ids[opposite.prompt_length:]
            difference = next(i for i, (a, b) in enumerate(zip(targets, alternate)) if a != b)
            prefix = response.input_ids[:response.prompt_length + difference]
            positive = model(torch.tensor([prefix], device=device), memories=memories,
                             life_id=episode.episode_id)[0][:, -1]
            negative = model(torch.tensor([prefix], device=device), memories=other_memories,
                             life_id=other_episode.episode_id)[0][:, -1]
            probabilities = [F.log_softmax(model.head(h).float(), -1)[0, targets[difference]]
                             for h in (positive, negative)]
            contrast = F.softplus(2.0 - (probabilities[0] - probabilities[1]))
            total += 2 * len(prefix)
        losses.append(answer + cf_weight * contrast)
        answers.append(float(answer.detach()))
        contrasts.append(float(contrast.detach()))
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite SHINE loss")
    return loss, total, dict(answer_loss=sum(answers) / len(answers),
                              counterfactual_loss=sum(contrasts) / len(contrasts), queries=len(losses))


def episode_token_cost(model, compiler, row, cf_weight, direct=False):
    episode = compiler.compile(row)
    n = 0 if direct else sum(len(s.input_ids) + model.memory_tokens for s in episode.source) * (2 if cf_weight else 1)
    n += sum(len(r.input_ids) for r in episode.responses)
    if cf_weight:
        other = compiler.compile(shine_episode(row["annotations"]["ordinal"],
                 variant=1 - row["annotations"]["variant"], task=row["annotations"]["payload_task"],
                 seed=row["annotations"]["stateless_seed"], language=row["language"], split=row["split"]))
        for a, b in zip(episode.responses, other.responses):
            dif = next(i for i, (x, y) in enumerate(zip(a.input_ids[a.prompt_length:], b.input_ids[b.prompt_length:])) if x != y)
            n += 2 * (a.prompt_length + dif)
    return n


@torch.no_grad()
def evaluate_fixture(model, compiler, rows, *, direct=None, views=("memory", "off"),
                     journal_path=None, identity=None, resume=False, ledger=None, pause_check=None):
    saved_rng = capture_rng(model.base.embed_tokens.weight.device)
    was_training = model.training
    model.eval()
    records = []
    jobs=[f"{number}/{p['probe_id']}/{view}" for number,row in enumerate(rows) for p in row["probes"] for view in views]
    journal=EvaluationJournal(journal_path,identity,jobs,resume=resume) if journal_path else None
    try:
        for number, row in enumerate(rows):
            episode = compiler.compile(row)
            memories=None
            for probe, response in zip(row["probes"], episode.responses):
                for view in views:
                    if pause_check and pause_check():
                        if journal:journal.pause_requested=True
                        else:raise EvaluationPaused("SHINE fixture evaluation paused")
                    def compute():
                        nonlocal memories
                        source_cost=0
                        query=evidence_teacher_response(compiler,row,probe) if view=="plaintext" else response
                        if view=="memory" and memories is None:
                            predicted=0 if direct else sum(len(s.input_ids)+model.memory_tokens for s in episode.source)
                            charge=ledger.reserve(run=str(journal_path),kind="evaluation_source",student=predicted) if ledger else None
                            memories,source_cost=(encode_memories(model,episode) if direct is None else (direct.memories(number,row),0))
                            if ledger:ledger.commit(charge,student=source_cost)
                        used=memories if view=="memory" else []
                        upper=len(query.prompt_ids)+16
                        charge=ledger.reserve(run=str(journal_path),kind="evaluation_query",
                              student=upper if view=="memory" else 0,teacher=0 if view=="memory" else upper) if ledger else None
                        result=model.generate(query.prompt_ids,memories=used,life_id=episode.episode_id,
                                              eos_token_id=compiler.tokenizer.eos_token_id)
                        if ledger:ledger.commit(charge,student=result["input_tokens"] if view=="memory" else 0,
                                                teacher=0 if view=="memory" else result["input_tokens"])
                        text=compiler.tokenizer.decode(result["token_ids"],skip_special_tokens=True)
                        return dict(context_index=number, episode_id=row["episode_id"], probe_id=probe["probe_id"],
                                        view=view, expected=probe["target"], text=text,
                                        exact_match=normalized_answer(text) == normalized_answer(probe["target"]),
                                        exact_token_match=result["stopped_on_end"] and result["token_ids"]+[compiler.tokenizer.eos_token_id]==query.input_ids[query.prompt_length:],
                                        source_input_tokens=source_cost,**result)
                    key=f"{number}/{probe['probe_id']}/{view}"
                    records.append(journal.evaluate(key,compute) if journal else compute())
        summary = {view: dict(queries=sum(r["view"] == view for r in records),
                             correct=sum(r["view"] == view and r["exact_match"] for r in records),
                             exact_tokens=sum(r["view"]==view and r["exact_token_match"] for r in records),
                             ended=sum(r["view"] == view and r["stopped_on_end"] for r in records)) for view in views}
        if journal:journal.finish()
        return dict(summary=summary, records=records, diagnostic_only=True,
                    evaluation_query_input_tokens=sum(r["input_tokens"] for r in records),
                    source_encoding_input_tokens=sum(r["source_input_tokens"] for r in records),
                    student_input_tokens=sum(r["source_input_tokens"]+(r["input_tokens"] if r["view"]=="memory" else 0) for r in records),
                    teacher_input_tokens=sum(r["input_tokens"] for r in records if r["view"]!="memory"))
    finally:
        model.train(was_training)
        restore_rng(saved_rng, model.base.embed_tokens.weight.device)


def main():
    import argparse
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", choices=["oracle", "generator"], default="generator")
    parser.add_argument("--task", choices=["U", "P", "C"], default="U")
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--grad-accum", type=int, default=6)
    parser.add_argument("--student-tokens", type=int, required=True)
    parser.add_argument("--max-micro-steps", type=int, default=1000000)
    parser.add_argument("--cf-weight", type=float, default=1.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--evaluate-every", type=int, default=0)
    parser.add_argument("--preflight-plaintext", action="store_true")
    parser.add_argument("--diagnostic-ledger", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.worlds < 1 or args.student_tokens < 1 or args.grad_accum < 1 or args.max_micro_steps < 0
            or args.lr <= 0 or args.cf_weight < 0 or not np.isfinite(args.lr + args.cf_weight)
            or args.evaluate_only and not args.resume):
        raise ValueError("invalid SHINE diagnostic arguments")
    root, run = Path(args.base).resolve(), Path(args.run_dir).resolve()
    device, _ = resolve_device(args.device)
    ledger_path=Path(args.diagnostic_ledger).resolve()
    if device.startswith("cuda") and any(not p.is_relative_to(Path("/mnt/e/ds-rsi")) for p in (root, run,ledger_path)):
        raise ValueError("SHINE CUDA growth paths must remain on E:")
    if device.startswith("cuda") and torch.cuda.mem_get_info()[0]<8*(1<<30):
        raise RuntimeError("SHINE preflight requires 8GiB available device memory")
    ledger=DiagnosticBudget(ledger_path)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = ShineQwen.from_local_checkpoint(root, device=device)
    rows = fixture_rows(worlds=args.worlds, task=args.task, seed=args.seed)
    compiler = EpisodeCompiler(AutoTokenizer.from_pretrained(root, local_files_only=True))
    direct = DirectFactors(model, rows).to(device) if args.mode == "oracle" else None
    if direct is not None:
        model.requires_grad_(False)
    trainable_model = direct if direct is not None else model
    parameters = [p for p in trainable_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0., foreach=False)
    source_root = Path(__file__).resolve().parents[3]
    config = {k: v for k, v in vars(args).items() if k not in {"base", "run_dir", "resume", "initialize_only",
               "evaluate_only", "evaluate_every", "max_micro_steps", "preflight_plaintext"}}
    weights = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    from dsrsi.model.cuda_execution import cuda_execution_identity, require_reference_kernels
    if device.startswith("cuda"):
        require_reference_kernels(model)
    identity = json.loads(json.dumps(dict(train_config=config, architecture=model.architecture_identity(),
        base=dict(tokenizer=tokenizer_fingerprint(root), weights={n:file_sha256(root/n) for n in sorted(set(weights.values()))}),
        data=dict(version=SHINE_DATA_VERSION, fixture=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()),
        implementation={str(p.relative_to(source_root)):file_sha256(p) for p in sorted((source_root/"src/dsrsi").rglob("*.py"))},
        script_sha256=file_sha256(source_root/"scripts/shine_pilot.py"),
        runtime=cuda_execution_identity(device) if device.startswith("cuda") else {"torch":str(torch.__version__)})))
    store = CheckpointStore(run/"checkpoints")
    control = ControlChannel(run)
    control.install_signal_handlers()
    progress = Progress("SHINE_diagnostic")
    if args.resume:
        payload, manifest = store.load_latest(identity=identity)
        progress, cursor, life = restore_training_state(payload, trainable_model, optimizer, device=device)
        if cursor != {"next_ordinal":progress.micro_step} or life:
            raise ValueError("SHINE diagnostic cursor/life differs")
    elif (store.root/"latest.json").exists():
        raise FileExistsError("SHINE run exists; explicit compatible resume required")
    events = run/"events.jsonl"
    def log(value):
        with events.open("a") as f:
            f.write(json.dumps(value, allow_nan=False)+"\n")
            f.flush()
    def save(status):
        if any(s.mode != "off" or s.active for s in model._sites.values()):
            raise RuntimeError("active generated tensors at checkpoint boundary")
        payload = capture_training_state(trainable_model, optimizer, progress=progress,
                     data_cursor={"next_ordinal":progress.micro_step}, life_state={}, device=device)
        d=store.begin(progress)
        store.save_rank(d,payload)
        committed=store.commit(d,world_size=1,identity=identity,progress=progress,status=status)
        log(dict(event="saved",status=status,directory=committed.name,progress=asdict(progress)))
        return committed
    if not args.resume:
        save("stopped")
    print(json.dumps(dict(event="ready",progress=asdict(progress),trainable_parameters=sum(p.numel() for p in parameters),
                          architecture=model.architecture_identity())),flush=True)
    if args.initialize_only:
        return
    def pause_evaluation():
        request=control.poll()
        if request and request.action=="checkpoint":
            control.acknowledge(request,save("checkpoint"))
            return False
        return request is not None
    def evaluate(output,views):
        if output.exists():return json.loads(output.read_text())
        weight_hash=hashlib.sha256()
        for name,param in trainable_model.named_parameters():
            if param.requires_grad:
                weight_hash.update(name.encode())
                weight_hash.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        journal=Path(str(output)+".journal.json")
        evaluation_identity=dict(training_identity=identity,micro=progress.micro_step,
                                  weights_sha256=weight_hash.hexdigest(),views=list(views))
        try:
            result=evaluate_fixture(model,compiler,rows,direct=direct,views=views,journal_path=journal,
                       identity=evaluation_identity,resume=args.resume and journal.exists(),ledger=ledger,pause_check=pause_evaluation)
        except EvaluationPaused:
            request=control.poll()
            if request is None:raise
            control.acknowledge(request,save("paused" if request.action=="pause" else "stopped"))
            raise
        atomic_json(output,result)
        return result
    if args.preflight_plaintext:
        out=run/f"preflight-micro{progress.micro_step}.json"
        result=evaluate(out,("plaintext","off"))
        print(json.dumps(dict(event="plaintext_preflight",summary=result["summary"])),flush=True)
    if args.evaluate_only:
        out=run/f"evaluation-micro{progress.micro_step}.json"
        evaluate(out,("memory","off"))
        return
    existing=[json.loads(l) for l in events.read_text().splitlines()]
    micros=[x for x in existing if x["event"]=="micro"]
    if len(micros)!=progress.micro_step:
        raise RuntimeError("event/checkpoint disagreement; inspect without restarting")
    compute_seconds=sum(r["micro_seconds"] for r in micros)
    if args.evaluate_every and progress.micro_step and progress.micro_step%args.evaluate_every==0:
        out=run/f"evaluation-micro{progress.micro_step}.json"
        if Path(str(out)+".journal.json").exists():evaluate(out,("memory","off"))
    stop_reason="micro_limit"
    while progress.micro_step < args.max_micro_steps:
        request=control.poll()
        if request:
            d=save({"pause":"paused","stop":"stopped","checkpoint":"checkpoint"}[request.action])
            control.acknowledge(request,d)
            if request.action!="checkpoint":return
        index=(progress.micro_step*7)%len(rows)
        row=rows[index]
        cost=episode_token_cost(model,compiler,row,args.cf_weight,direct is not None)
        if progress.student_tokens+cost>args.student_tokens:
            stop_reason="diagnostic_student_budget"
            break
        try:
            charge=ledger.reserve(run=str(run),kind="training_micro",student=cost)
        except DiagnosticBudgetExceeded:
            stop_reason="shared_diagnostic_budget";break
        start=time.monotonic()
        counter=rows[index^1]
        loss,tokens,statistics=objective(model,compiler,row,direct=direct,context_index=index,
                                       counterfactual=counter,cf_weight=args.cf_weight)
        if tokens!=cost:raise RuntimeError("actual SHINE token cost differs from admission check")
        (loss/args.grad_accum).backward()
        accumulation=progress.accumulation_index+1
        step=progress.optimizer_step
        if accumulation==args.grad_accum:
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step+=1
            accumulation=0
        if device.startswith("cuda"):torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        ledger.commit(charge,student=tokens)
        compute_seconds+=elapsed
        progress=Progress("SHINE_diagnostic",step,progress.micro_step+1,accumulation,progress.student_tokens+tokens,0)
        record=dict(event="micro",episode_id=row["episode_id"],loss=float(loss.detach()),student_input_tokens=tokens,
                    micro_seconds=elapsed,progress=asdict(progress),**statistics)
        if device.startswith("cuda"):
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        log(record)
        if progress.micro_step%20==0:save("checkpoint")
        if args.evaluate_every and progress.micro_step%args.evaluate_every==0:
            output=run/f"evaluation-micro{progress.micro_step}.json"
            try:result=evaluate(output,("memory","off"))
            except DiagnosticBudgetExceeded:
                stop_reason="shared_diagnostic_budget";break
            print(json.dumps(dict(event="fixture_evaluation",micro=progress.micro_step,summary=result["summary"])),flush=True)
        if compute_seconds>=3*3600:
            stop_reason="training_wall_budget";break
        if device.startswith("cuda") and (torch.cuda.max_memory_allocated()>6*(1<<30)
                                            or torch.cuda.max_memory_reserved()>7*(1<<30)):
            stop_reason="memory_guard";break
        import shutil
        if device.startswith("cuda") and shutil.disk_usage(run).free<20*(1<<30):
            stop_reason="disk_guard";break
    d=save("stopped")
    summary=dict(status="stopped",reason=stop_reason,checkpoint=d.name,manifest_sha256=file_sha256(d/"manifest.json"),
                 progress=asdict(progress),compute_seconds=compute_seconds,diagnostic_only=True,
                 A_formal_tokens_consumed=0,trainable_parameters=sum(p.numel() for p in parameters))
    atomic_json(run/f"training-summary-micro{progress.micro_step}.json",summary)
    print(json.dumps(summary),flush=True)
