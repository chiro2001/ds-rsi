#!/usr/bin/env python3
"""Fetch a revision-pinned Wikipedia corpus for held-out language retention.

Text is taken from ``action=parse&oldid=<revid>`` so every document is pinned
to an immutable revision and can be re-fetched byte-for-byte with the same
converter.  Runs on the development host, not inside the training container.
The corpus itself stays outside Git; only the manifest is committed.
"""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

HOSTS = {"en": "https://en.wikipedia.org", "zh": "https://zh.wikipedia.org"}
USER_AGENT = "ds-rsi-language-retention/1.0 (research corpus fetch; contact: chiro2001)"
LICENSE = "CC BY-SA 4.0"


def api(language: str, parameters: dict, *, timeout: float):
    query = urllib.parse.urlencode(parameters)
    request = urllib.request.Request(f"{HOSTS[language]}/w/api.php?{query}",
                                     headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(15 * (2 ** attempt))
    raise RuntimeError("unreachable retry state")


def html_to_text(text: str) -> str:
    """Deterministic HTML-to-plain-text conversion for pinned revisions."""
    body = re.sub(r"(?is)<(script|style|sup|table|figure)[^>]*>.*?</\1>", " ", text)
    body = re.sub(r"(?is)<br\s*/?>", "\n", body)
    body = re.sub(r"(?is)</(p|div|li|h[1-6]|dd|dt|tr|blockquote)>", "\n", body)
    body = re.sub(r"(?s)<[^>]+>", "", body)
    body = html.unescape(body)
    body = re.sub(r"[ \t\u00a0\u200b]+", " ", body)
    body = re.sub(r"\n\s*\n\s*\n+", "\n\n", body)
    return "\n".join(line.strip() for line in body.splitlines()).strip()


def truncate(text: str, maximum: int) -> str:
    text = text.strip()
    if len(text) <= maximum:
        return text
    head = text[:maximum]
    for marker in ("\n\n", "。", ". "):
        index = head.rfind(marker)
        if index >= int(maximum * 0.6):
            return head[:index + len(marker)].strip()
    return head.strip()


def latest_revision(language: str, title: str, *, timeout: float) -> tuple[str, int]:
    payload = api(language, {"action": "query", "format": "json", "formatversion": "2",
                             "prop": "revisions", "rvprop": "ids", "redirects": "1",
                             "titles": title}, timeout=timeout)
    pages = payload.get("query", {}).get("pages")
    if not pages or "missing" in pages[0] or not pages[0].get("revisions"):
        raise RuntimeError(f"missing article {language}/{title}")
    return pages[0]["title"], int(pages[0]["revisions"][0]["revid"])


def render_revision(language: str, revid: int, *, maximum: int, timeout: float) -> tuple[str, str]:
    payload = api(language, {"action": "parse", "format": "json", "formatversion": "2",
                             "oldid": str(revid), "prop": "text"}, timeout=timeout)
    parsed = payload.get("parse")
    if not parsed or int(parsed.get("revid", 0)) != revid:
        raise RuntimeError(f"revision {language}/{revid} did not round-trip")
    return parsed["title"], truncate(html_to_text(parsed["text"]), maximum)


def slugify(title: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", title).strip("-").lower()
    if not slug:
        raise ValueError("title produces an empty slug")
    return slug


def corpus_body(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def build_manifest(rows, *, corpus_path, titles_path, maximum, schema, file_sha256, text_sha256):
    return {"schema": schema, "source": "Wikipedia rendered revisions (MediaWiki parse API)",
            "license": LICENSE, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_file": str(corpus_path), "corpus_sha256": file_sha256(corpus_path),
            "title_list": str(titles_path), "title_list_sha256": file_sha256(titles_path),
            "max_characters": maximum, "converter": "html-to-text-v1",
            "documents": [{"document_id": row["document_id"], "language": row["language"],
                           "title": row["title"], "revision_id": row["revision_id"],
                           "url": row["url"], "license": row["license"],
                           "characters": len(row["text"]), "sha256": text_sha256(row["text"])}
                          for row in rows]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles", default="configs/broad-language-titles-v1.json")
    parser.add_argument("--out", default="data/broad-language/corpus-v1.jsonl")
    parser.add_argument("--manifest", default="data/manifests/broad-language-corpus-v1.json")
    parser.add_argument("--max-characters", type=int, default=6000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--verify", action="store_true",
                        help="re-render every pinned revision and compare hashes; writes nothing")
    args = parser.parse_args()
    if args.max_characters < 2000:
        parser.error("at least 2000 characters are required per document")
    from dsrsi.data.broad_language import CORPUS_REVISION, file_sha256, text_sha256
    manifest_path = Path(args.manifest)
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != CORPUS_REVISION:
            parser.error("unexpected corpus manifest schema")
        mismatches, verified = [], 0
        for entry in manifest["documents"]:
            title, text = render_revision(entry["language"], entry["revision_id"],
                                          maximum=manifest["max_characters"], timeout=args.timeout)
            matches = text_sha256(text) == entry["sha256"] and title == entry["title"]
            verified += 1
            if not matches:
                mismatches.append(entry["document_id"])
            print(f"verified {entry['document_id']} revision={entry['revision_id']} "
                  + ("ok" if matches else "MISMATCH"), flush=True)
            time.sleep(args.delay)
        if mismatches:
            raise SystemExit(f"revision mismatch: {mismatches}")
        print(json.dumps({"verified": verified, "status": "all_revisions_reproduced"}))
        return
    titles = json.loads(Path(args.titles).read_text())
    if titles.get("revision") != "broad-language-titles-v1":
        parser.error("unexpected title list revision")
    rows = []
    for language in ("en", "zh"):
        for title in titles["languages"][language]:
            resolved, revid = latest_revision(language, title, timeout=args.timeout)
            rendered_title, text = render_revision(language, revid, maximum=args.max_characters,
                                                   timeout=args.timeout)
            rows.append({"document_id": f"{language}-{slugify(resolved)}", "language": language,
                         "title": rendered_title, "revision_id": revid,
                         "url": f"{HOSTS[language]}/w/index.php?oldid={revid}",
                         "license": LICENSE, "text": text})
            print(f"fetched {rows[-1]['document_id']} rev={revid} chars={len(text)}", flush=True)
            time.sleep(args.delay)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus_body(rows))
    manifest = build_manifest(rows, corpus_path=output, titles_path=Path(args.titles),
                              maximum=args.max_characters, schema=CORPUS_REVISION,
                              file_sha256=file_sha256, text_sha256=text_sha256)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"documents": len(rows), "corpus_sha256": manifest["corpus_sha256"],
                      "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
