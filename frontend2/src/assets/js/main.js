const backendHost = "10.49.12.47:9999";
const REST_URL = `https://${backendHost}/api/offer`;
const WS_USER = `wss://${backendHost}/ws/video_user`;
const WS_RL = `wss://${backendHost}/ws/video_rl`;
const WS_ACTION = `wss://${backendHost}/ws/ws`;

const video_user = document.getElementById("video_user");
const video_rl = document.getElementById("video_rl");
const loading_user = document.getElementById("loading_user");
const loading_rl = document.getElementById("loading_rl");

let ws_user, ws_rl, ws_action, session_id;

const token = localStorage.getItem('token');

// Send an offer to the backend to start a new session
async function startSession() {
    const response = await fetch(REST_URL, { method: "POST", headers: { "Authorization": `Bearer ${token}`} });
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
        // Hide loading overlay on first frame
        if (loading_user.style.display !== "none") {
            loading_user.style.display = "none";
        }
    };
    ws_user.onclose = ws_user.onerror = () => {
        loading_user.style.display = "flex";
    };

    ws_rl = new WebSocket(`${WS_RL}?session_id=${session_id}`);
    ws_rl.binaryType = "arraybuffer";
    ws_rl.onmessage = (event) => {
        const blob = new Blob([event.data], { type: "image/jpeg" });
        video_rl.src = URL.createObjectURL(blob);
        // Hide loading overlay on first frame
        if (loading_rl.style.display !== "none") {
            loading_rl.style.display = "none";
        }
    };
    ws_rl.onclose = ws_rl.onerror = () => {
        loading_rl.style.display = "flex";
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
    switch (e.key.toLowerCase()) {  
        case "a": 
            a[0] = -1.0;
            break;
        case "d": 
            a[0] = +1.0;
            break;
        case "w": 
            a[1] = +1.0;
            break;
        case "s": 
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
    switch (e.key.toLowerCase()) {  
        case "a": 
            a[0] = -1.0;
            break;
        case "d": 
            a[0] = +1.0;
            break;
        case "w": 
            a[1] = +1.0;
            break;
        case "s": 
            a[2] = +0.8;
            break;
        default:
            return; 
    }
    if (ws_action && ws_action.readyState === WebSocket.OPEN) {
        ws_action.send(JSON.stringify({ action: a }));
    }
}

window.addEventListener("load", async () => {
    // Always show loading overlays at start
    loading_user.style.display = "flex";
    loading_rl.style.display = "flex";
    await startSession();
    startVideoStreams();
    startActionWebSocket();
});

//refesh the page whent the user clicks on the refresh button
document.getElementById("startBtn").addEventListener("click", () => {
    location.reload();
});