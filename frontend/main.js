const backendHost = "localhost:443";
const REST_URL = `https://${backendHost}/api/offer`;
const WS_USER = `wss://${backendHost}/video_user`;
const WS_RL = `wss://${backendHost}/video_rl`;
const WS_ACTION = `wss://${backendHost}/ws`;

const video_user = document.getElementById("video_user");
const video_rl = document.getElementById("video_rl");
const loading_user = document.getElementById("loading_user");
const loading_rl = document.getElementById("loading_rl");

let ws_user, ws_rl, ws_action, session_id;

// Send an offer to the backend to start a new session
async function startSession() {
    const response = await fetch(REST_URL, { method: "POST" });
    const data = await response.json();
    session_id = data.session_id;
}

function startVideoStreams() {
    if (!session_id) {
        console.error("Cannot open WebSocket: no session_id yet");
        return;
    }
    ws_user = new WebSocket(`${WS_USER}?session_id=${session_id}`);
    ws_user.binaryType = "arraybuffer";
    ws_user.onmessage = (event) => {
        const blob = new Blob([event.data], { type: "image/jpeg" });
        video_user.src = URL.createObjectURL(blob);
    };

    ws_rl = new WebSocket(`${WS_RL}?session_id=${session_id}`);
    ws_rl.binaryType = "arraybuffer";
    ws_rl.onmessage = (event) => {
        const blob = new Blob([event.data], { type: "image/jpeg" });
        video_rl.src = URL.createObjectURL(blob);
    };
}

function startActionWebSocket() {
    if (!session_id) {
        console.error("Cannot open WebSocket: no session_id yet");
        return;
    }
    ws_action = new WebSocket(`${WS_ACTION}?session_id=${session_id}`);
    ws_action.onopen = () => {
        document.addEventListener("keydown", onKeyDownSendAction);
        document.addEventListener("keyup", onKeyUpSendAction);
    };
}

let a = [0.0, 0.0, 0.0];
function onKeyDownSendAction(e) {
    switch (e.key) {
        case "ArrowLeft":
            a[0] = -1.0;
            break;
        case "ArrowRight":
            a[0] = +1.0;
            break;
        case "ArrowUp":
            a[1] = +1.0;
            break;
        case "ArrowDown":
            a[2] = +0.8;
            break;
        default:
            return;
    }
    if (ws_action && ws_action.readyState === WebSocket.OPEN) {
        ws_action.send(JSON.stringify({ action: a }));
    }
}
function onKeyUpSendAction(e) {
    switch (e.key) {
        case "ArrowLeft":
        case "ArrowRight":
            a[0] = 0.0;
            break;
        case "ArrowUp":
            a[1] = 0.0;
            break;
        case "ArrowDown":
            a[2] = 0.0;
            break;
        default:
            return;
    }
    if (ws_action && ws_action.readyState === WebSocket.OPEN) {
        ws_action.send(JSON.stringify({ action: a }));
    }
}

window.addEventListener("load", () => {
    startVideoStreams();
    startActionWebSocket();
});
