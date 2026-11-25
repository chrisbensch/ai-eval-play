"""Task contracts and runner-owned tests for the coding benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    title: str
    image: str
    filename: str
    languages: tuple[str, ...]
    prompt: str
    grader_filename: str
    grader: str
    cases: tuple[str, ...]
    compile_command: str
    test_command: str
    build_seconds: int = 45
    scratch_size: str = "96m"


PYTHON_TEST_MAIN = """
if __name__ == "__main__":
    from pathlib import Path

    result = unittest.main(verbosity=2, exit=False).result
    if result.testsRun != 1 or result.skipped:
        sys.exit(2)
    if not result.wasSuccessful():
        sys.exit(1)
    Path(".case-passed").touch()
"""

HTTP_TEST_SUPPORT = """
import http.client
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path


class UI(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = set()
        self.forms = {}
        self.script = ""
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        if tag == "form":
            attributes = dict(attrs)
            self.forms[attributes.get("action")] = (attributes.get("method") or "GET").upper()
        self.in_script |= tag == "script"

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.script += data


class HTTPTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = str(Path(self.tmp.name) / "tickets.sqlite")
        self.start()

    def stop(self):
        if self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()

    def start(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.server = subprocess.Popen(
            [sys.executable, "-B", "app.py", "--host", "127.0.0.1",
             "--port", str(self.port), "--db", self.db],
            stdin=subprocess.DEVNULL,
        )
        self.addCleanup(self.stop)
        for _ in range(100):
            if self.server.poll() is not None:
                self.fail("app.py exited before serving HTTP")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return
            except OSError:
                time.sleep(0.025)
        self.fail("app.py did not start listening")

    def request(self, method, path, tenant="a", body=None, key=None, *,
                extra_headers=None, include_headers=False):
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if tenant is not None:
            headers["X-Tenant"] = tenant
        if key is not None:
            headers["Idempotency-Key"] = key
        if body is not None and not isinstance(body, str):
            body = json.dumps(body)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            conn.request(method, path, body=body, headers=headers)
            result = conn.getresponse()
            text = result.read().decode()
            if result.getheader("Content-Type", "").startswith("application/json"):
                text = json.loads(text)
            if include_headers:
                return result.status, text, dict(result.getheaders())
            return result.status, text
        finally:
            conn.close()

"""


WEB = Task(
    title="Web app",
    image="python:3.12-slim",
    filename="app.py",
    languages=("python", "py"),
    grader_filename="grader.py",
    prompt="""Implement a Python 3.12 standard-library-only issue tracker as one app.py.
It must serve a browser UI as well as an HTTP JSON API, use SQLite, and run with
python app.py --host 127.0.0.1 --port PORT --db PATH. Bind that host/port.
Use Content-Type application/json for API responses. Endpoints:

* GET / returns HTML with a title, a form with a label for its ticket-title
  input, and a ticket list. The page must load tickets and POST new ones
  through the API using inline browser JavaScript. Use demo as the tenant in
  the UI. Render user text safely and show API failures in the page.
* Every /api/tickets endpoint requires a nonblank X-Tenant header (400 if
  omitted). A tenant cannot list, read, or modify another tenant's tickets.
* POST /api/tickets accepts {"title":"..."} and requires Idempotency-Key.
  In a transaction, create a ticket and return 201 with
  {"id":<globally unique positive integer>,"title":"...","version":1}. An exact
  replay for the same tenant and key returns 200 and the original body. Reusing the
  key with a different title returns 409. The same key in another tenant is
  independent. Reject blank titles or keys, absent/malformed fields, and
  malformed JSON with 400 and do not consume the key.
* GET /api/tickets/ID returns a ticket object, or 404 if missing in this tenant.
* PATCH /api/tickets/ID accepts {"title":"...","version":<positive integer>}.
  If the version matches, atomically update the title, increment the version,
  and return 200 and the updated object. For a stale version return 409 with
  no mutation; missing/foreign ticket is 404 and invalid input is 400.
* GET /api/tickets?limit=N&after=ID returns {"items":[ticket,...],
  "next_after":integer-or-null}. Sort by ascending IDs greater than after.
  Defaults: limit=20 and after=0. Accept limits 1–100; invalid values are 400.
  Return the last included ID as next_after only if more tenant rows exist.

Handle concurrent requests correctly and preserve tickets and idempotency data
on restart. Use no external network or packages. Return only one fenced Python
block containing the complete app.py. Omit test code and explanations.""",
    grader=HTTP_TEST_SUPPORT + """
class Tests(HTTPTestCase):
    def add(self, title, key, tenant="a"):
        return self.request("POST", "/api/tickets", tenant, {"title": title}, key)

    def test_web_page(self):
        status, page = self.request("GET", "/", tenant=None)
        self.assertEqual(status, 200)
        ui = UI()
        ui.feed(page)
        self.assertTrue({"title", "form", "label", "input", "script"} <= ui.tags)
        self.assertIn("/api/tickets", ui.script)
        self.assertIn("X-Tenant", ui.script)
        self.assertIn("POST", ui.script)
        status, _ = self.request("POST", "/api/tickets", body="{broken", key="x")
        self.assertEqual(status, 400)
        self.assertEqual(self.add("First", "x")[0], 201)

    def test_idempotency(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: self.add("Same", "x"), range(3)))
        self.assertEqual(sorted(code for code, _ in results), [200, 200, 201])
        self.assertEqual(results[0][1], results[1][1])
        self.assertEqual(results[0][1], results[2][1])
        self.assertEqual(self.add("Changed", "x")[0], 409)
        self.assertEqual(len(self.request("GET", "/api/tickets")[1]["items"]), 1)

    def test_tenant_isolation(self):
        a = self.add("A", "same-key")[1]
        b = self.add("B", "same-key", "b")[1]
        self.assertNotEqual(a["id"], b["id"])
        target = f'/api/tickets/{a["id"]}'
        self.assertEqual(self.request("GET", target, "b")[0], 404)
        self.assertEqual(self.request("PATCH", target, "b", {"title":"bad","version":1})[0], 404)
        self.assertEqual(self.request("GET", "/api/tickets", None)[0], 400)
        self.assertEqual(self.request("GET", "/api/tickets", "b")[1]["items"], [b])

    def test_version_conflict(self):
        original = self.add("Original", "key")[1]
        target = f'/api/tickets/{original["id"]}'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda title: self.request("PATCH", target, body={"title":title,"version":1}),
                ("Left", "Right"),
            ))
        self.assertEqual(sorted(code for code, _ in results), [200, 409])
        winner = next(body for code, body in results if code == 200)
        self.assertEqual(winner["version"], 2)
        self.assertEqual(self.request("GET", target), (200, winner))
        self.assertEqual(self.add("Original", "key"), (200, original))

    def test_pagination_and_restart(self):
        a = self.add("A", "a")[1]
        self.add("Foreign", "x", "b")
        b = self.add("B", "b")[1]
        self.stop()
        self.start()
        code, page = self.request("GET", "/api/tickets?limit=1")
        self.assertEqual(code, 200)
        self.assertEqual(page, {"items":[a], "next_after":a["id"]})
        self.assertEqual(
            self.request("GET", f'/api/tickets?limit=1&after={a["id"]}'),
            (200, {"items":[b], "next_after":None}),
        )
        self.assertEqual(self.add("A", "a"), (200, a))
        self.assertEqual(self.request("GET", "/api/tickets?limit=0")[0], 400)
""" + PYTHON_TEST_MAIN,
    cases=("test_web_page", "test_idempotency", "test_tenant_isolation",
           "test_version_conflict", "test_pagination_and_restart"),
    compile_command="python -B -m py_compile app.py grader.py",
    test_command='python -B grader.py "Tests.$case_name"',
)


RUST = Task(
    title="Rust",
    image="rust:1.85-bookworm",
    filename="journal.rs",
    languages=("rust", "rs"),
    grader_filename="grader.rs",
    prompt="""Implement a durable append-only binary journal in one Rust 2021 module journal.rs.
Use only std and no unsafe, external crates, main function, or tests. Expose:
pub fn append(path: &std::path::Path, data: &[u8]) -> std::io::Result<()>
pub fn recover(path: &std::path::Path) -> std::io::Result<Vec<Vec<u8>>>

A frame is a little-endian u32 payload length, then a little-endian u32 FNV-1a
checksum of the payload, then the payload. FNV-1a starts at 2166136261 and
for each byte performs XOR followed by wrapping multiply by 16777619.
Empty frames are valid. Accept at most 1,048,576 bytes per frame.
Missing file recovers as an empty journal without creating a file.
Recovery validates complete frames in order. An incomplete final header or
payload is a torn tail: truncate to the previous frame boundary and sync the
repair before returning. A checksum mismatch on a complete frame or a length
over the limit in a complete header is InvalidData; leave the entire file
unchanged in that case. An oversized append is InvalidInput and does not
create or modify the file. A valid append validates and repairs a torn tail
before writing, leaves corrupt files unchanged, creates a missing file, and
syncs after writing before returning success. Callers serialize access to the
same path. Keep I/O and allocation bounded per frame while reading.
Return only one fenced Rust block with the complete module.""",
    grader="""
#![forbid(unsafe_code)]
#[path = "journal.rs"]
mod journal;
use std::{fs, io::ErrorKind, path::Path};

fn completed() {
    fs::write(".case-passed", b"").unwrap();
}

fn frame(payload: &[u8]) -> Vec<u8> {
    let checksum = payload.iter().fold(2166136261_u32, |hash, &byte| {
        (hash ^ u32::from(byte)).wrapping_mul(16777619)
    });
    let mut raw = (payload.len() as u32).to_le_bytes().to_vec();
    raw.extend(checksum.to_le_bytes());
    raw.extend(payload);
    raw
}

#[test]
fn test_wire_format() {
    let path = Path::new("wire.log");
    assert!(journal::recover(path).unwrap().is_empty());
    assert!(!path.exists());
    journal::append(path, b"hello").unwrap();
    journal::append(path, b"").unwrap();
    journal::append(path, &[0, 1, 255]).unwrap();
    let expected = [frame(b"hello"), frame(b""), frame(&[0, 1, 255])].concat();
    assert_eq!(fs::read(path).unwrap(), expected);
    assert_eq!(journal::recover(path).unwrap(), vec![b"hello".to_vec(), vec![], vec![0, 1, 255]]);
    completed();
}

#[test]
fn test_torn_header() {
    let path = Path::new("header.log");
    let first = frame(b"safe");
    fs::write(path, [first.clone(), vec![2, 0, 0]].concat()).unwrap();
    assert_eq!(journal::recover(path).unwrap(), vec![b"safe".to_vec()]);
    assert_eq!(fs::read(path).unwrap(), first);
    completed();
}

#[test]
fn test_torn_payload_then_append() {
    let path = Path::new("payload.log");
    let second = frame(b"interrupted");
    fs::write(path, [frame(b"first"), second[..10].to_vec()].concat()).unwrap();
    journal::append(path, b"after").unwrap();
    assert_eq!(fs::read(path).unwrap(), [frame(b"first"), frame(b"after")].concat());
    assert_eq!(journal::recover(path).unwrap(), vec![b"first".to_vec(), b"after".to_vec()]);
    completed();
}

#[test]
fn test_corruption_preserved() {
    let path = Path::new("corrupt.log");
    let mut bytes = [frame(b"valid"), frame(b"broken")].concat();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(path, &bytes).unwrap();
    assert_eq!(journal::recover(path).unwrap_err().kind(), ErrorKind::InvalidData);
    assert_eq!(journal::append(path, b"add").unwrap_err().kind(), ErrorKind::InvalidData);
    assert_eq!(fs::read(path).unwrap(), bytes);
    completed();
}

#[test]
fn test_size_limits() {
    let path = Path::new("limit.log");
    let max = vec![0xA5; 1_048_576];
    let too_big = vec![0; 1_048_577];
    assert_eq!(journal::append(path, &too_big).unwrap_err().kind(), ErrorKind::InvalidInput);
    assert!(!path.exists());
    journal::append(path, &max).unwrap();
    assert_eq!(journal::recover(path).unwrap(), vec![max]);
    let header = [1_048_577_u32.to_le_bytes(), 0_u32.to_le_bytes()].concat();
    fs::write(path, &header).unwrap();
    assert_eq!(journal::recover(path).unwrap_err().kind(), ErrorKind::InvalidData);
    assert_eq!(fs::read(path).unwrap(), header);
    completed();
}
""",
    cases=("test_wire_format", "test_torn_header", "test_torn_payload_then_append",
           "test_corruption_preserved", "test_size_limits"),
    compile_command="rustc --edition=2021 --test grader.rs -o grade",
    test_command='./grade --exact "$case_name" --nocapture',
)


CPP = Task(
    title="C++20",
    image="rust:1.85-bookworm",
    filename="channel.hpp",
    languages=("cpp", "c++", "cxx", "hpp", "cpp20", "c++20"),
    grader_filename="grader.cpp",
    prompt="""Implement a bounded multi-producer multi-consumer channel as a single C++20
header channel.hpp. Use only the standard library supported by GCC 12. In the
global namespace:
  template<class T> class BoundedChannel
  explicit BoundedChannel(std::size_t capacity);
  bool push(T value);
  std::optional<T> pop();
  bool push_for(T value, std::chrono::milliseconds timeout);
  std::optional<T> pop_for(std::chrono::milliseconds timeout);
  void close();

Reject capacity zero with std::invalid_argument. Preserve FIFO order, support
move-only T, and use blocking synchronization rather than sleep/poll loops.
An open push waits for space; an open pop waits for data. Timed calls return
false/nullopt after the supplied timeout when they cannot proceed. Close is
idempotent, wakes all blocked producers and consumers, rejects all further
pushes, and allows queued items to drain before pop returns nullopt. All
operations may race with close. Callers join threads before destruction.
Return only one fenced C++ code block containing the complete header. Omit
main, tests, and explanations.""",
    grader="""
#include "channel.hpp"
#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <future>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
using namespace std::chrono_literals;

static void test_fifo_and_move_only() {
    bool threw = false;
    try { BoundedChannel<int> invalid(0); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    BoundedChannel<std::unique_ptr<int>> q(2);
    assert(q.push(std::make_unique<int>(10)));
    assert(q.push(std::make_unique<int>(20)));
    auto first = q.pop();
    auto second = q.pop();
    assert(first && second && **first == 10 && **second == 20);
}

static void test_blocking_and_capacity() {
    BoundedChannel<int> q(1);
    assert(q.push(7));
    std::promise<void> entered;
    auto started = entered.get_future();
    auto producer = std::async(std::launch::async, [&] { entered.set_value(); return q.push(8); });
    started.wait();
    assert(producer.wait_for(60ms) == std::future_status::timeout);
    assert(q.pop() == 7);
    assert(producer.get());
    assert(q.pop() == 8);
}

static void test_timed_operations() {
    BoundedChannel<int> q(1);
    auto start = std::chrono::steady_clock::now();
    assert(!q.pop_for(60ms));
    assert(std::chrono::steady_clock::now() - start >= 45ms);
    assert(q.push_for(3, 0ms));
    start = std::chrono::steady_clock::now();
    assert(!q.push_for(4, 60ms));
    assert(std::chrono::steady_clock::now() - start >= 45ms);
    assert(q.pop_for(0ms) == 3);
}

static void test_close_wakes_and_drains() {
    BoundedChannel<int> full(1);
    assert(full.push(5));
    std::promise<void> entered;
    auto started = entered.get_future();
    auto blocked = std::async(std::launch::async, [&] { entered.set_value(); return full.push(6); });
    started.wait();
    assert(blocked.wait_for(40ms) == std::future_status::timeout);
    full.close();
    full.close();
    assert(!blocked.get());
    assert(full.pop() == 5);
    assert(!full.pop());
    assert(!full.push_for(9, 0ms));

    BoundedChannel<int> empty(1);
    std::promise<void> consuming;
    auto waiting = consuming.get_future();
    auto consumer = std::async(std::launch::async, [&] { consuming.set_value(); return empty.pop(); });
    waiting.wait();
    assert(consumer.wait_for(40ms) == std::future_status::timeout);
    empty.close();
    assert(!consumer.get());
}

static void test_multi_producer_consumer() {
    BoundedChannel<int> q(4);
    std::vector<int> left, right;
    std::thread a([&] { while (auto v = q.pop()) left.push_back(*v); });
    std::thread b([&] { while (auto v = q.pop()) right.push_back(*v); });
    std::thread c([&] { for (int i = 0; i < 200; ++i) assert(q.push(i)); });
    std::thread d([&] { for (int i = 200; i < 400; ++i) assert(q.push(i)); });
    c.join();
    d.join();
    q.close();
    a.join();
    b.join();
    left.insert(left.end(), right.begin(), right.end());
    std::sort(left.begin(), left.end());
    assert(left.size() == 400);
    for (int i = 0; i < 400; ++i) assert(left[i] == i);
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string_view name = argv[1];
    if (name == "test_fifo_and_move_only") test_fifo_and_move_only();
    else if (name == "test_blocking_and_capacity") test_blocking_and_capacity();
    else if (name == "test_timed_operations") test_timed_operations();
    else if (name == "test_close_wakes_and_drains") test_close_wakes_and_drains();
    else if (name == "test_multi_producer_consumer") test_multi_producer_consumer();
    else return 2;
    std::ofstream completed(".case-passed");
    if (!completed) return 3;
}
""",
    cases=("test_fifo_and_move_only", "test_blocking_and_capacity", "test_timed_operations",
           "test_close_wakes_and_drains", "test_multi_producer_consumer"),
    compile_command="g++ -std=c++20 -O2 -pthread grader.cpp -o grade",
    test_command='./grade "$case_name"',
)


AGENT_REQUIREMENTS = """A maintenance ticket asks you to finish a small Python invoice-import repository.
Inspect it, edit the relevant source files, and run the public tests before finishing.
Use only the Python 3.12 standard library. Preserve the function signatures and CLI:
  python -m invoices.cli DB_PATH CSV_PATH

CSV has the exact header invoice_id,customer,amount. Strip surrounding whitespace
from values. IDs and customers must be nonblank. Amounts are nonnegative decimal
strings with one or more whole-number digits and optionally one or two fractional
digits. Convert exactly to integer cents; do not use binary floating point. Reject
malformed rows, invalid headers, and blank required fields with ValueError.

ingest(db_path, csv_text) returns {"inserted": N, "duplicates": N}. Reimporting an
identical invoice ID, customer and cents is a no-op duplicate, including within a
single batch. Same ID with a different customer or cents raises ValueError. Validate
and persist each import atomically: conflicting or invalid rows must leave all
previous data intact and insert none of the new batch.

totals(db_path) returns an insertion-ordered dict mapping customers to total cents,
sorted by customer name, and returns {} for a missing or empty database. On success
the CLI emits one JSON object {"inserted": N, "duplicates": N, "totals": {...}} and
exits 0. On input, file, or database errors it writes a concise stderr message and
exits nonzero without a traceback. Keep this a small patch to the existing project.
The invoice files and tests are tracked in the virtual repo. Only invoices/parser.py,
invoices/store.py, and invoices/cli.py are editable.
"""

AGENT_FILES = {
    "README.md": """# Invoice importer

Import a batch with `python -m invoices.cli DB_PATH CSV_PATH`.
Run the public checks with `python -m unittest discover -s tests -v`.
The parser handles input normalization, the store owns SQLite, and the CLI
formats the import result.
""",
    "invoices/__init__.py": "",
    "invoices/parser.py": """
import csv
import io


def parse_rows(csv_text):
    reader = csv.DictReader(io.StringIO(csv_text))
    return [
        (row["invoice_id"], row["customer"], int(float(row["amount"]) * 100))
        for row in reader
    ]
""",
    "invoices/store.py": """
import sqlite3

from .parser import parse_rows


def ingest(db_path, csv_text):
    rows = parse_rows(csv_text)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS invoices "
            "(invoice_id TEXT PRIMARY KEY, customer TEXT, cents INTEGER)"
        )
        conn.executemany("INSERT INTO invoices VALUES (?, ?, ?)", rows)
    return {"inserted": len(rows), "duplicates": 0}


def totals(db_path):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT customer, SUM(cents) FROM invoices GROUP BY customer")
        return dict(rows)
""",
    "invoices/cli.py": """
import argparse
import json
from pathlib import Path

from .store import ingest, totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    result = ingest(args.db_path, args.csv_path.read_text(encoding="utf-8"))
    result["totals"] = totals(args.db_path)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
    "tests/test_public.py": """
import tempfile
import unittest
from pathlib import Path

from invoices.store import ingest, totals

HEADER = "invoice_id,customer,amount\\n"


class PublicTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = Path(folder.name) / "invoices.db"

    def test_import_and_total(self):
        result = ingest(self.db, HEADER + "A,Alice,1.25\\n")
        self.assertEqual(result, {"inserted": 1, "duplicates": 0})
        self.assertEqual(totals(self.db), {"Alice": 125})

    def test_identical_reimport(self):
        data = HEADER + "B,Bob,3.00\\n"
        ingest(self.db, data)
        self.assertEqual(ingest(self.db, data), {"inserted": 0, "duplicates": 1})

    def test_invalid_amount(self):
        with self.assertRaises(ValueError):
            ingest(self.db, HEADER + "C,Carol,-1.00\\n")
""",
}

AGENT_EDITABLE = frozenset(("invoices/parser.py", "invoices/store.py", "invoices/cli.py"))

AGENT = Task(
    title="Agentic maintenance",
    image="python:3.12-slim",
    filename="",
    languages=(),
    prompt=AGENT_REQUIREMENTS,
    grader_filename="grader.py",
    grader="""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from invoices.store import ingest, totals

HEADER = "invoice_id,customer,amount\\n"


class Tests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        self.db = self.directory / "invoices.db"

    def test_exact_cents_and_sorted_totals(self):
        self.assertEqual(totals(self.db), {})
        data = HEADER + " a , Zed ,1.15\\nb,Zed,0.29\\nc,Ada,12\\nd,Ada,0\\n"
        self.assertEqual(ingest(self.db, data), {"inserted":4, "duplicates":0})
        result = totals(self.db)
        self.assertEqual(result, {"Ada":1200, "Zed":144})
        self.assertEqual(list(result), ["Ada", "Zed"])

    def test_duplicate_policy(self):
        before = HEADER + "x,Lee,2.00\\nx,Lee,2\\n"
        self.assertEqual(ingest(self.db, before), {"inserted":1, "duplicates":1})
        self.assertEqual(ingest(self.db, before), {"inserted":0, "duplicates":2})
        with self.assertRaises(ValueError):
            ingest(self.db, HEADER + "x,Other,2\\n")
        with self.assertRaises(ValueError):
            ingest(self.db, HEADER + "x,Lee,2.01\\n")
        self.assertEqual(totals(self.db), {"Lee":200})

    def test_atomic_rollback(self):
        ingest(self.db, HEADER + "existing,Ada,2\\n")
        for bad in ("existing,Ada,3\\n", "late,Bob,-3.00\\n"):
            with self.assertRaises(ValueError):
                ingest(self.db, HEADER + "new,Ada,5\\n" + bad)
            self.assertEqual(totals(self.db), {"Ada":200})
        self.assertEqual(ingest(self.db, HEADER + "new,Ada,5\\n")["inserted"], 1)

    def test_input_validation(self):
        for data in ("bad,header\\n", HEADER + "a,Ada,1.234\\n", HEADER + ",Ada,1\\n",
                     HEADER + "a,   ,1\\n", HEADER + "a,Ada,1e2\\n",
                     HEADER + "a,Ada,1,extra\\n", HEADER + "a,Ada\\n"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                ingest(self.db, data)

    def test_cli(self):
        csv_file = self.directory / "batch.csv"
        csv_file.write_text(HEADER + "ok,Ada,2.50\\n", encoding="utf-8")
        args = [sys.executable, "-B", "-m", "invoices.cli", str(self.db), str(csv_file)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {"inserted":1, "duplicates":0, "totals":{"Ada":250}})
        csv_file.write_text(HEADER + "bad,Ada,-1\\n", encoding="utf-8")
        result = subprocess.run(args, capture_output=True, text=True, timeout=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.strip())
        self.assertNotIn("Traceback", result.stderr)
""" + PYTHON_TEST_MAIN,
    cases=("test_exact_cents_and_sorted_totals", "test_duplicate_policy",
           "test_atomic_rollback", "test_input_validation", "test_cli"),
    compile_command="python -B -m py_compile invoices/*.py grader.py",
    test_command='python -B grader.py "Tests.$case_name"',
)

PYTHON_WEB = Task(
    title="Python reservation app",
    image="python:3.12-slim",
    filename="app.py",
    languages=("python", "py"),
    grader_filename="grader.py",
    prompt="""Build a complete Python 3.12 stock-reservation web app as one app.py. Use only
the standard library and SQLite. Run as:
  python app.py --host 127.0.0.1 --port PORT --db PATH

Sessions and roles
- Seed these local demo accounts: alice/alice-demo and bob/bob-demo have role
  member; admin/admin-demo has role admin. Use salted password hashes and
  unpredictable session tokens. A login has its own session.
- POST /api/login accepts {"username":"...","password":"..."} and returns 200
  with {"username":"...","role":"member" or "admin"} and a cookie named
  session. Include HttpOnly, SameSite=Lax and Path=/ on the cookie. Reject
  bad credentials with 401. POST /api/logout revokes the session and returns
  200. Other /api routes require a valid session (401 if absent or revoked).
- POST /api/items is admin-only (403 for members). Accept
  {"sku":"...","name":"...","quantity":positive integer}. Return 201 with
  {"sku":"...","name":"...","available":quantity}, or 409 for an existing SKU.
  GET /api/items returns {"items":[item,...]} sorted by SKU.

Reservations and history
- POST /api/reservations accepts {"sku":"...","quantity":positive integer}
  and requires a nonblank Idempotency-Key header. Atomically reserve stock
  and create a reservation. Return 201 with {"id":positive integer,
  "sku":"...","quantity":integer,"status":"active"}. Unknown SKU is 404;
  insufficient stock is 409. Replaying identical input for the same user
  and key returns 200 and the original creation object, even after
  cancellation. Reusing a key with different input returns 409.
- GET /api/reservations returns {"reservations":[reservation,...]} for the
  signed-in user only, in ID order. Allow POST /api/reservations/ID/cancel
  by the owner. Return 404 for a missing or foreign ID. Return 200 and the
  reservation with status "cancelled". A repeat returns the same result
  and must not restore stock twice or append another event.
- GET /api/activity returns {"events":[{"kind":"reserved" or "cancelled",
  "reservation_id":integer},...]} in creation order, scoped to the caller.
  Commit each inventory change, its reservation, and its event together.
  Preserve stock, reservations, idempotency keys and history across restarts.
- JSON endpoints use application/json. Return 400 for malformed JSON or
  fields, blank strings or non-positive/non-integer quantities, including
  booleans. Make simultaneous requests safe.

Browser workflow
- GET / renders an HTML page titled Stock Reservations. Signed-out visitors
  get a labeled login form POSTing to /login with username and password.
  Signed-in visitors get a dashboard with available stock, their reservations,
  and a labeled form POSTing sku and quantity to /reserve.
- Successful form POSTs accept application/x-www-form-urlencoded and redirect
  to / with 303. Login sets the same session cookie as API login. The reserve
  form uses the same transactional inventory logic. Render user data safely
  and show useful errors.

Return only one fenced Python block. Include the entire app, with no tests,
external dependencies, or network requests to other hosts.""",
    grader=HTTP_TEST_SUPPORT + """
from http.cookies import SimpleCookie
from urllib.parse import urlencode


class Tests(HTTPTestCase):
    def call(self, method, path, cookie=None, body=None, key=None, **options):
        headers = options.pop("extra_headers", {})
        if cookie:
            headers = {"Cookie": cookie, **headers}
        return self.request(method, path, tenant=None, body=body, key=key,
                            extra_headers=headers, **options)

    def extract_cookie(self, headers):
        raw = next((v for k, v in headers.items() if k.lower() == "set-cookie"), "")
        cookies = SimpleCookie()
        cookies.load(raw)
        self.assertIn("session", cookies)
        session = cookies["session"]
        self.assertTrue(session.value and session["httponly"])
        self.assertEqual(session["path"], "/")
        self.assertEqual(session["samesite"].lower(), "lax")
        return f"session={session.coded_value}"

    def login(self, username):
        status, data, headers = self.call(
            "POST", "/api/login", body={"username":username, "password":username+"-demo"},
            include_headers=True,
        )
        self.assertEqual(status, 200)
        self.assertEqual(data, {"username":username, "role":"admin" if username == "admin" else "member"})
        return self.extract_cookie(headers)

    def add_stock(self, qty=4):
        admin = self.login("admin")
        self.assertEqual(self.call("POST", "/api/items", admin, {"sku":"w1","name":"Wrench","quantity":qty}),
                         (201, {"sku":"w1", "name":"Wrench", "available":qty}))
        return admin

    def reserve(self, cookie, qty=2, key="x"):
        return self.call("POST", "/api/reservations", cookie, {"sku":"w1","quantity":qty}, key)

    def available(self, cookie):
        status, data = self.call("GET", "/api/items", cookie)
        self.assertEqual(status, 200)
        return data["items"][0]["available"]

    def test_login_roles_and_revocation(self):
        self.assertEqual(self.call("GET", "/api/items")[0], 401)
        self.assertEqual(self.call("POST", "/api/login", body={"username":"alice","password":"wrong"})[0], 401)
        alice = self.login("alice")
        other_session = self.login("alice")
        self.assertNotEqual(alice, other_session)
        self.assertEqual(self.call("POST", "/api/items", alice, {"sku":"w1","name":"Wrench","quantity":3})[0], 403)
        self.add_stock(3)
        self.assertEqual(self.call("POST", "/api/logout", alice)[0], 200)
        self.assertEqual(self.call("GET", "/api/items", alice)[0], 401)
        self.assertEqual(self.available(other_session), 3)

    def test_concurrent_stock(self):
        self.add_stock(5)
        members = [self.login("alice"), self.login("bob")]
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = list(pool.map(lambda cookie: self.reserve(cookie, 3), members))
        self.assertEqual(sorted(status for status, _ in answers), [201, 409])
        self.assertEqual(self.available(members[0]), 2)
        for cookie, (status, body) in zip(members, answers):
            reservations = self.call("GET", "/api/reservations", cookie)[1]["reservations"]
            self.assertEqual(reservations, [body] if status == 201 else [])

    def test_idempotency_cancellation_and_ownership(self):
        self.add_stock()
        alice, bob = self.login("alice"), self.login("bob")
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda _: self.reserve(alice), range(2)))
        self.assertEqual(sorted(code for code, _ in replies), [200, 201])
        initial = replies[0][1]
        self.assertEqual(replies[1][1], initial)
        self.assertEqual(self.reserve(alice, 1)[0], 409)
        self.assertEqual(self.call("GET", "/api/reservations", bob)[1]["reservations"], [])
        target = f'/api/reservations/{initial["id"]}/cancel'
        self.assertEqual(self.call("POST", target, bob)[0], 404)
        cancelled = dict(initial, status="cancelled")
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda _: self.call("POST", target, alice), range(2)))
        self.assertEqual(replies, [(200, cancelled), (200, cancelled)])
        self.assertEqual(self.reserve(alice), (200, initial))
        self.assertEqual(self.available(bob), 4)
        self.assertEqual(self.call("GET", "/api/activity", alice), (200, {"events": [
            {"kind":"reserved", "reservation_id":initial["id"]},
            {"kind":"cancelled", "reservation_id":initial["id"]},
        ]}))
        self.assertEqual(self.reserve(bob, 1)[0], 201)

    def test_persistent_history_and_validation(self):
        self.add_stock()
        alice = self.login("alice")
        for body in ('{unclosed', {"sku":"w1", "quantity":True}, {"sku":"w1", "quantity":0}):
            self.assertEqual(self.call("POST", "/api/reservations", alice, body, "k")[0], 400)
        status, initial = self.reserve(alice, 3, "k")
        self.assertEqual(status, 201)
        target = f'/api/reservations/{initial["id"]}/cancel'
        self.assertEqual(self.call("POST", target, alice)[0], 200)
        self.stop()
        self.start()
        alice, bob = self.login("alice"), self.login("bob")
        expected = [{"kind":kind, "reservation_id":initial["id"]} for kind in ("reserved","cancelled")]
        self.assertEqual(self.call("GET", "/api/activity", alice), (200, {"events":expected}))
        self.assertEqual(self.call("GET", "/api/activity", bob), (200, {"events":[]}))
        self.assertEqual(self.available(alice), 4)
        self.assertEqual(self.reserve(alice, 3, "k"), (200, initial))

    def test_browser_form_workflow(self):
        self.add_stock()
        status, page = self.call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Stock Reservations", page)
        ui = UI()
        ui.feed(page)
        self.assertEqual(ui.forms.get("/login"), "POST")
        self.assertIn("label", ui.tags)
        form = {"Content-Type": "application/x-www-form-urlencoded"}
        status, _, headers = self.call("POST", "/login",
            body=urlencode({"username":"alice","password":"alice-demo"}), extra_headers=form, include_headers=True)
        self.assertEqual(status, 303)
        self.assertEqual(next(v for k, v in headers.items() if k.lower() == "location"), "/")
        alice = self.extract_cookie(headers)
        status, dashboard = self.call("GET", "/", alice)
        self.assertEqual(status, 200)
        self.assertIn("Wrench", dashboard)
        ui = UI()
        ui.feed(dashboard)
        self.assertEqual(ui.forms.get("/reserve"), "POST")
        status, _, headers = self.call("POST", "/reserve", alice,
            body=urlencode({"sku":"w1","quantity":2}), extra_headers=form, include_headers=True)
        self.assertEqual(status, 303)
        self.assertEqual(next(v for k, v in headers.items() if k.lower() == "location"), "/")
        self.assertEqual(self.available(alice), 2)
        self.assertEqual(len(self.call("GET", "/api/reservations", alice)[1]["reservations"]), 1)
""" + PYTHON_TEST_MAIN,
    cases=("test_login_roles_and_revocation", "test_concurrent_stock",
           "test_idempotency_cancellation_and_ownership", "test_persistent_history_and_validation",
           "test_browser_form_workflow"),
    compile_command="python -B -m py_compile app.py grader.py",
    test_command='python -B grader.py "Tests.$case_name"',
)


GO = Task(
    title="Go",
    image="golang:1.25.14-bookworm",
    filename="cache.go",
    languages=("go", "golang"),
    grader_filename="grader_test.go",
    prompt="""Implement a concurrent loading cache in a single Go 1.25 file, cache.go, in
package cache. Use only the standard library. Public API:

type Loader func(context.Context, string) (string, error)
func New(ttl time.Duration, now func() time.Time, load Loader) *Cache
func (c *Cache) Get(ctx context.Context, key string) (string, error)
func (c *Cache) Invalidate(key string)

Cache successful results per key for ttl, measured from when the load
completes. An entry expires as soon as now() reaches the deadline. A
non-positive ttl disables storage but still coalesces concurrent requests.
Use the provided clock, or time.Now if now is nil. Errors never enter the cache.

Concurrent callers for the same missing key share a single load and its
result. Different keys must be able to load concurrently. An already-cancelled
caller returns ctx.Err() without using cached data or starting a load. A caller
that cancels while waiting returns ctx.Err() promptly. The shared load runs with
a context independent of the caller's cancellation. A successful load may still
be cached after all original callers have stopped waiting.

Invalidate removes an existing value and detaches any in-flight load. Callers
already waiting on a detached load can receive its result. Callers arriving after
Invalidate must start or join a new load, and the detached result must never
overwrite or clear the new cache entry or its in-flight state. All public methods
must be safe under concurrent use. The runner builds with Go's race detector.
Return only one fenced Go block with the complete implementation. Omit main,
module metadata and tests.""",
    grader=r"""
package cache_test

import (
    "benchmark.local/cache"
    "context"
    "errors"
    "os"
    "sync"
    "sync/atomic"
    "testing"
    "time"
)

type result struct {
    value string
    err error
}

func finish(t *testing.T) {
    t.Helper()
    t.Cleanup(func() {
        if !t.Failed() && !t.Skipped() {
            if err := os.WriteFile(".case-passed", nil, 0600); err != nil {
                t.Error(err)
            }
        }
    })
}

func receive[T any](t *testing.T, ch <-chan T) T {
    t.Helper()
    select {
    case v := <-ch:
        return v
    case <-time.After(time.Second):
        t.Fatal("timed out waiting for cache operation")
        var zero T
        return zero
    }
}

func startGet(c *cache.Cache, ctx context.Context, key string) <-chan result {
    reply := make(chan result, 1)
    go func() { v, err := c.Get(ctx, key); reply <- result{v, err} }()
    return reply
}

func expect(t *testing.T, result result, value string, err error) {
    t.Helper()
    if result.value != value || !errors.Is(result.err, err) {
        t.Fatalf("got (%q, %v), want (%q, %v)", result.value, result.err, value, err)
    }
}

func TestCoalescesAndLoadsKeysIndependently(t *testing.T) {
    finish(t)
    started := make(chan string, 24)
    release := make(chan struct{})
    c := cache.New(time.Minute, nil, func(_ context.Context, key string) (string,error) {
        started <- key
        <-release
        return "value-"+key, nil
    })
    ctx := context.Background()
    var ready sync.WaitGroup
    replies := make([]<-chan result, 0, 12)
    ready.Add(12)
    launch := make(chan struct{})
    for range 12 {
        reply := make(chan result, 1)
        replies = append(replies, reply)
        go func() {
            ready.Done()
            <-launch
            v, err := c.Get(ctx, "a")
            reply <- result{v, err}
        }()
    }
    ready.Wait()
    close(launch)
    if key := receive(t, started); key != "a" { t.Fatal(key) }
    b := startGet(c, ctx, "b")
    if key := receive(t, started); key != "b" { t.Fatalf("extra load for %s before key b", key) }
    time.Sleep(35*time.Millisecond)
    close(release)
    expect(t, receive(t, b), "value-b", nil)
    for _, reply := range replies { expect(t, receive(t, reply), "value-a", nil) }
    select {
    case key := <-started: t.Fatalf("unexpected extra load for %s", key)
    default:
    }
}

func TestTTLMeasuredFromCompletion(t *testing.T) {
    finish(t)
    var clock, count atomic.Int64
    now := func() time.Time { return time.Unix(0, clock.Load()) }
    started, release := make(chan struct{}), make(chan struct{})
    c := cache.New(10*time.Millisecond, now, func(context.Context,string) (string,error) {
        if count.Add(1) == 1 { close(started); <-release }
        return "ok", nil
    })
    ctx := context.Background()
    first := startGet(c, ctx, "x")
    receive(t, started)
    clock.Store(int64(50*time.Millisecond))
    close(release)
    expect(t, receive(t, first), "ok", nil)
    clock.Store(int64(59*time.Millisecond))
    expect(t, receive(t, startGet(c, ctx, "x")), "ok", nil)
    if count.Load() != 1 { t.Fatal("expired early") }
    clock.Store(int64(60*time.Millisecond))
    expect(t, receive(t, startGet(c, ctx, "x")), "ok", nil)
    if count.Load() != 2 { t.Fatal("did not expire at exact deadline") }
}

func TestErrorsAndDisabledStorage(t *testing.T) {
    finish(t)
    var calls atomic.Int64
    broken := errors.New("load failed")
    ctx := context.Background()
    c := cache.New(time.Minute, nil, func(context.Context,string) (string,error) {
        if calls.Add(1) == 1 { return "", broken }
        return "recovered", nil
    })
    expect(t, receive(t, startGet(c, ctx, "x")), "", broken)
    expect(t, receive(t, startGet(c, ctx, "x")), "recovered", nil)
    expect(t, receive(t, startGet(c, ctx, "x")), "recovered", nil)
    if calls.Load() != 2 { t.Fatal("failed or successful result stored incorrectly") }
    started, release := make(chan struct{}), make(chan struct{})
    uncached := cache.New(0, nil, func(context.Context,string) (string,error) {
        if calls.Add(1) == 3 { close(started) }
        <-release
        return "uncached", nil
    })
    first := startGet(uncached, ctx, "x")
    receive(t, started)
    other := startGet(uncached, ctx, "x")
    time.Sleep(35*time.Millisecond)
    close(release)
    expect(t, receive(t, first), "uncached", nil)
    expect(t, receive(t, other), "uncached", nil)
    if calls.Load() != 3 { t.Fatal("zero TTL did not coalesce concurrent loads") }
    expect(t, receive(t, startGet(uncached, ctx, "x")), "uncached", nil)
    if calls.Load() != 4 { t.Fatal("zero TTL stored a value") }
}

func TestCancelledWaiterLeavesSharedLoad(t *testing.T) {
    finish(t)
    started, release := make(chan struct{}), make(chan struct{})
    var calls atomic.Int64
    c := cache.New(time.Minute, nil, func(ctx context.Context, _ string) (string,error) {
        if calls.Add(1) == 1 { close(started) }
        select {
        case <-ctx.Done(): return "", ctx.Err()
        case <-release: return "ready", nil
        }
    })
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    first := startGet(c, ctx, "x")
    receive(t, started)
    other := startGet(c, context.Background(), "x")
    cancel()
    expect(t, receive(t, first), "", context.Canceled)
    close(release)
    expect(t, receive(t, other), "ready", nil)
    expect(t, receive(t, startGet(c, ctx, "x")), "", context.Canceled)
    expect(t, receive(t, startGet(c, ctx, "new-key")), "", context.Canceled)
    if calls.Load() != 1 { t.Fatal("shared loader was lost after cancellation") }
}

func TestInvalidateDetachesInflight(t *testing.T) {
    finish(t)
    started, release := make(chan struct{}), make(chan struct{})
    freshStarted, releaseFresh := make(chan struct{}), make(chan struct{})
    var calls atomic.Int64
    c := cache.New(time.Minute, nil, func(context.Context,string) (string,error) {
        switch calls.Add(1) {
        case 1:
            close(started)
            <-release
            return "old", nil
        case 2:
            close(freshStarted)
            <-releaseFresh
        }
        return "fresh", nil
    })
    ctx := context.Background()
    old := startGet(c, ctx, "x")
    receive(t, started)
    c.Invalidate("x")
    fresh := startGet(c, ctx, "x")
    receive(t, freshStarted)
    close(release)
    expect(t, receive(t, old), "old", nil)
    another := startGet(c, ctx, "x")
    select {
    case reply := <-another: t.Fatalf("returned while replacement load was pending: %+v", reply)
    case <-time.After(35*time.Millisecond):
    }
    close(releaseFresh)
    expect(t, receive(t, fresh), "fresh", nil)
    expect(t, receive(t, another), "fresh", nil)
    expect(t, receive(t, startGet(c, ctx, "x")), "fresh", nil)
    if calls.Load() != 2 { t.Fatal("detached result invalidated the new cached value") }
    c.Invalidate("x")
    expect(t, receive(t, startGet(c, ctx, "x")), "fresh", nil)
    if calls.Load() != 3 { t.Fatal("invalidate did not clear an existing value") }
}
""",
    cases=("TestCoalescesAndLoadsKeysIndependently", "TestTTLMeasuredFromCompletion",
           "TestErrorsAndDisabledStorage", "TestCancelledWaiterLeavesSharedLoad",
           "TestInvalidateDetachesInflight"),
    compile_command=(
        "mkdir -p source && mv cache.go grader_test.go source/ && cd source && "
        "printf 'module benchmark.local/cache\\n\\ngo 1.25.0\\n' > go.mod && "
        "env GOCACHE=/tmp/go-cache GOPATH=/tmp/go GOTOOLCHAIN=local GOPROXY=off "
        "GOSUMDB=off GOWORK=off GOMAXPROCS=2 CGO_ENABLED=1 go test -race -c -o /tmp/grade ."
    ),
    test_command='./grade -test.run="^$case_name$" -test.timeout=6s',
    build_seconds=150,
    scratch_size="320m",
)


TASKS = {
    "web": WEB,
    "python-web": PYTHON_WEB,
    "rust": RUST,
    "cpp": CPP,
    "go": GO,
    "agent": AGENT,
}
