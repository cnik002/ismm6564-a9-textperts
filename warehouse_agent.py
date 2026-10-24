"""The warehouse team's agent, reached over A2A (version 1.0: JSON-RPC over HTTP).

Written by the Assignment 9 notebook from the team's own functions. It runs as its own program:
    python warehouse_agent.py 8765
and answers at http://127.0.0.1:8765/ (its card is at /.well-known/agent-card.json). It reads its settings from
environment variables: WAREHOUSE_DB (its own database), WAREHOUSE_MODEL and OLLAMA_URL. It is told nothing about the
support system: not where the order database is, not who the customer is.
"""
import contextlib
import inspect
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai import OpenAI

import warehouse_api as wh

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
URL = f"http://127.0.0.1:{PORT}/"
MODEL = os.environ.get("WAREHOUSE_MODEL", "qwen2.5:7b")
llm = OpenAI(base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434/v1/"), api_key="ollama")
WAREHOUSE_API_TOOLS = ("check_stock", "hold_item", "release_hold")
TASK_STORE = {}
JSON_TYPE = {str: "string", int: "integer", float: "number", bool: "boolean"}


def model_chat(messages, tools=None):
    """One call to the warehouse agent's own model, through Ollama."""
    kw = {"tools": tools} if tools else {}
    r = llm.chat.completions.create(model=MODEL, messages=messages, temperature=0, seed=7, max_tokens=250, **kw)
    return r.choices[0].message, r.usage


WAREHOUSE_TOOL_NAMES = ['check_stock']      # the warehouse agent's tools
WAREHOUSE_SYSTEM = "You are the warehouse assistant for Northwind Outfitters. You check available inventory and stock levels. If a request asks for anything outside checking stock (such as starting returns, order details, refunds, customer info, policy questions, or changing arbitrary records), reply exactly 'OUT OF SCOPE: ' followed by a brief reason, and do not call any tools."


# ---- provided by the notebook
@contextlib.contextmanager
def wh_db():
    """A connection to the warehouse's database, for `with wh_db() as con:`. Changes are saved when the block ends."""
    con = sqlite3.connect(wh.DB)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def describe(fn, description=None):
    """The description of one tool, in the format the model API expects (from its name, docstring and arguments)."""
    params = inspect.signature(fn).parameters
    return {"type": "function",
            "function": {"name": fn.__name__,
                         "description": description or " ".join((inspect.getdoc(fn) or "").split()),
                         "parameters": {"type": "object",
                                        "properties": {name: {"type": JSON_TYPE.get(p.annotation, "string")} for name, p in params.items()},
                                        "required": [name for name, p in params.items() if p.default is inspect.Parameter.empty]}}}


def warehouse_tools(names):
    """The warehouse's functions with these names, by name. A name the warehouse does not offer a model is left out."""
    return {n: getattr(wh, n) for n in names if n in WAREHOUSE_API_TOOLS}


def warehouse_loop(text, chat_fn, tool_names, system, max_steps=6):
    """The warehouse agent's own loop (Week 8's): its own model, its own tools, nothing else.
    Returns (reply, steps, usage). The reply is "" if the loop stopped at its step limit."""
    tools = warehouse_tools(tool_names)
    descriptions = [describe(fn) for fn in tools.values()]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    usage, steps = {"model_calls": 0, "tokens_in": 0, "tokens_out": 0}, []
    for _ in range(max_steps):
        reply, used = chat_fn(messages, tools=descriptions)
        usage["model_calls"] += 1
        usage["tokens_in"] += used.prompt_tokens
        usage["tokens_out"] += used.completion_tokens
        if not reply.tool_calls:
            return (reply.content or "").strip(), steps, usage
        messages.append({"role": "assistant", "content": reply.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in reply.tool_calls]})
        for c in reply.tool_calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            if name not in tools:
                result = f"ERROR: there is no tool called {name}."
            elif not isinstance(args, dict):
                result = "ERROR: the arguments could not be read."
            else:
                try:
                    result = str(tools[name](**args))
                except Exception as exc:
                    result = f"ERROR: {name} failed: {type(exc).__name__}: {exc}"
            steps.append({"tool": name, "arguments": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
    return "", steps, usage


def make_handler(url, chat_fn):
    """The HTTP side of the A2A server: the card at the well-known address, and each JSON-RPC request (a POST)
    handed to handle_request. An error inside becomes JSON-RPC error -32603, never a dropped connection."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/.well-known/agent-card.json":
                try:
                    self._send(200, agent_card(url))
                except Exception as exc:
                    self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
            except ValueError:
                request = None
            try:
                response = handle_request(request, chat_fn)
            except Exception as exc:
                response = {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None,
                            "error": {"code": -32603, "message": f"internal error: {type(exc).__name__}: {exc}"}}
            self._send(200, response)

        def log_message(self, *args):
            pass

    return Handler


# ---- the team's imports, constants and functions


def agent_card(url: str) -> dict:
    """The warehouse agent's card: who it is, where to reach it (url), and what it can do."""
    return {
        "name": "Northwind Warehouse Agent",
        "description": "Specialist agent that checks product stock and inventory availability.",
        "version": "1.0",
        "supportedInterfaces": [
            {
                "url": url,
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "check_stock",
                "name": "Check Stock",
                "description": "Checks available on-hand stock for a specific product and variant (size or color).",
                "tags": ["inventory", "stock", "availability"],
                "examples": [
                    "Do you have the Ridgeline 2 tent in green?",
                    "Check stock for Wool hiking socks in size M.",
                ],
            }
        ],
    }


def send_message(params: dict, chat_fn) -> dict:
    """SendMessage: run the warehouse agent on the message, and return {"task": the task}. Never raises."""
    message = params.get("message") if isinstance(params, dict) else {}
    if not isinstance(message, dict):
        message = {}

    # Extract text from parts
    parts = message.get("parts") or []
    text = ""
    if isinstance(parts, list) and parts and isinstance(parts[0], dict):
        text = str(parts[0].get("text", "")).strip()

    # Context ID: preserve caller's or generate new
    context_id = message.get("contextId") or f"ctx-{uuid.uuid4().hex[:8]}"

    task_id = f"task-{uuid.uuid4().hex[:8]}"
    state = "TASK_STATE_COMPLETED"
    reply = ""
    steps = []
    usage = {"model_calls": 0, "tokens_in": 0, "tokens_out": 0}

    # Empty text is rejected without invoking the model
    if not text:
        state = "TASK_STATE_REJECTED"
        reply = "OUT OF SCOPE: empty request message."
    else:
        try:
            reply_text, steps_list, usage_dict = warehouse_loop(
                text, chat_fn, WAREHOUSE_TOOL_NAMES, WAREHOUSE_SYSTEM
            )
            reply = reply_text
            steps = steps_list
            usage = usage_dict

            if reply.startswith("OUT OF SCOPE"):
                state = "TASK_STATE_REJECTED"
            elif not reply:
                state = "TASK_STATE_FAILED"
            else:
                state = "TASK_STATE_COMPLETED"
        except Exception:
            state = "TASK_STATE_FAILED"
            reply = "An internal error occurred while processing the request."

    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state},
        "artifacts": [
            {
                "artifactId": f"art-{uuid.uuid4().hex[:8]}",
                "name": "reply",
                "parts": [{"text": reply}],
            }
        ],
        "metadata": {
            "model_calls": usage.get("model_calls", 0),
            "tokens_in": usage.get("tokens_in", 0),
            "tokens_out": usage.get("tokens_out", 0),
            "steps": steps,
        },
    }

    # In-memory store
    TASK_STORE[task_id] = task

    # Log to warehouse database
    try:
        with wh_db() as con:
            con.execute(
                "INSERT INTO a2a_tasks (task_id, context_id, request, state, reply, steps) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, context_id, text, state, reply, json.dumps(steps)),
            )
    except Exception:
        pass

    return {"task": task}


def handle_request(request, chat_fn) -> dict:
    """One JSON-RPC request in, one JSON-RPC response out. Never raises."""
    if not isinstance(request, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid Request"},
        }

    req_id = request.get("id")
    if request.get("jsonrpc") != "2.0" or not isinstance(
        request.get("method"), str
    ):
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32600, "message": "Invalid Request"},
        }

    method = request["method"]
    params = request.get("params")

    if method == "SendMessage":
        if not isinstance(params, dict) or "message" not in params:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params: missing message object",
                },
            }
        try:
            res = send_message(params, chat_fn)
            return {"jsonrpc": "2.0", "id": req_id, "result": res}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {type(exc).__name__}: {exc}",
                },
            }

    elif method == "GetTask":
        if not isinstance(params, dict) or "id" not in params:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params: missing task id",
                },
            }
        task_id = params["id"]
        if task_id in TASK_STORE:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": TASK_STORE[task_id],
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32001, "message": "Task not found"},
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": "Method not found"},
        }


if __name__ == "__main__":
    print(f"warehouse agent at {URL} (model {MODEL})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(URL, model_chat)).serve_forever()
