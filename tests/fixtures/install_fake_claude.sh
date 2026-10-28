# Installs a fake `claude` that prints Claude Code-style stream-json without calling any model.
# It echoes the token and its own user on purpose, so a run can check scrubbing and isolation.
mkdir -p ~/.local/bin
cat > ~/.local/bin/claude <<'FAKE'
#!/bin/bash
cat > /dev/null
user=$(id -un)
cat <<JSON
{"type":"system","subtype":"init","model":"fake-claude"}
{"type":"assistant","parent_tool_use_id":null,"message":{"id":"m1","model":"fake-claude","content":[{"type":"thinking","thinking":"Read the verifier first."},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"env; id -un"}}]}}
{"type":"user","parent_tool_use_id":null,"message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"CLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}\nuser=${user}","is_error":false}]}}
{"type":"assistant","parent_tool_use_id":null,"message":{"id":"m2","model":"fake-claude","content":[{"type":"text","text":"done"}]}}
{"type":"result","subtype":"success","is_error":false,"num_turns":2,"result":"done"}
JSON
FAKE
chmod +x ~/.local/bin/claude
