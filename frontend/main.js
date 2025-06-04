// --------------
// 1. Configuration
// --------------
const backendHost = "172.24.0.237:443"; // your backend address (with port)
const REST_URL = `https://${backendHost}/offer`;
const WS_ORIGIN = `wss://${backendHost}/ws`;

// References to two <video> tags and loaders in index.html:
const video_user = document.getElementById("video_user");
const video_rl = document.getElementById("video_rl");
const loading_user = document.getElementById("loading_user");
const loading_rl = document.getElementById("loading_rl");

// We’ll hold these at top‐level so they can be reused in event handlers:
let pc; // RTCPeerConnection
let ws; // WebSocket (for sending “action”)
let session_id; // the UUID we get back from /offer

// --------------
// 2. Create RTCPeerConnection and set up <video> handlers
// --------------
function setupPeerConnection() {
    pc = new RTCPeerConnection({
        iceServers: [
            // You can add TURN or STUN servers here if needed
            { urls: "stun:stun.l.google.com:19302" },
        ],
    });

    // We know our backend will send exactly two video tracks:
    //   • first track → user (human) CarRacing
    //   • second track → RL CarRacing
    let trackCount = 0;
    pc.ontrack = (evt) => {
        if (evt.track.kind !== "video") return;

        trackCount += 1;
        const stream = new MediaStream([evt.track]);

        if (trackCount === 1) {
            // First video track = user
            video_user.srcObject = stream;
            loading_user.style.display = "none";
        } else if (trackCount === 2) {
            // Second video track = RL
            video_rl.srcObject = stream;
            loading_rl.style.display = "none";
        } else {
            console.warn("Received more than 2 tracks – ignoring extras");
        }
    };

    // We explicitly tell the browser we only want to receive two "video" transceivers:
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("video", { direction: "recvonly" });
}

// --------------
// 3. negotiate(): createOffer() → wait ICE → POST to /offer → setAnswer → open WebSocket
// --------------
async function negotiate() {
    // a) Make sure PC is initialized
    setupPeerConnection();

    // b) Create an SDP offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // c) WAIT for ICE gathering to finish (so our SDP includes all candidates)
    await new Promise((resolve) => {
        if (pc.iceGatheringState === "complete") {
            resolve();
        } else {
            function checkState() {
                if (pc.iceGatheringState === "complete") {
                    pc.removeEventListener(
                        "icegatheringstatechange",
                        checkState
                    );
                    resolve();
                }
            }
            pc.addEventListener("icegatheringstatechange", checkState);
        }
    });

    // d) POST the completed localDescription to /offer
    const payload = {
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
    };

    let resp;
    try {
        resp = await fetch(REST_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
    } catch (err) {
        console.error("Failed to reach /offer:", err);
        return;
    }

    if (!resp.ok) {
        console.error("Server returned error:", await resp.text());
        return;
    }

    const answerJson = await resp.json();
    // answerJson === { sdp: "...", type: "answer", session_id: "<UUID>" }

    // e) Save the session_id for our WebSocket
    session_id = answerJson.session_id;
    console.log("Received session_id from server:", session_id);

    // f) Set the server’s SDP answer as our remote description
    await pc.setRemoteDescription({
        sdp: answerJson.sdp,
        type: answerJson.type,
    });

    // g) Now that we have a valid session_id, open the WebSocket:
    openActionWebSocket();
}

// --------------
// 4. openActionWebSocket(): connect to ws://.../ws?session_id=<UUID>
// --------------
function openActionWebSocket() {
    if (!session_id) {
        console.error("Cannot open WebSocket: no session_id yet");
        return;
    }

    ws = new WebSocket(`${WS_ORIGIN}?session_id=${session_id}`);

    ws.addEventListener("open", () => {
        console.log(`WebSocket opened (session_id=${session_id})`);
    });

    ws.addEventListener("message", (evt) => {
        // Our server does not send us any data back on this channel,
        // but if you ever want ACKs or logging, handle it here:
        console.log("WS ←", evt.data);
    });

    ws.addEventListener("close", (evt) => {
        console.warn(`WebSocket closed (code=${evt.code})`);
    });

    ws.addEventListener("error", (err) => {
        console.error("WebSocket error:", err);
    });

    // Hook arrow‐key events to send { action: … } JSON over this WebSocket:
    document.addEventListener("keydown", onKeyDownSendAction);
    document.addEventListener("keyup", onKeyUpSendAction);
}

// --------------
// 5. Key handlers: send { action: … } or { action: 0 } over the WS
// --------------
// 'a' is  action vector:
//   a[0] = left/right   (−1.0 for left, +1.0 for right, 0 otherwise)
//   a[1] = throttle     (+1.0 when up is held, 0 otherwise)
//   a[2] = brake        (+0.8 when down is held, 0 otherwise)
let a = [0.0, 0.0, 0.0];

// Prevent arrow keys from scrolling the page
window.addEventListener(
    "keydown",
    function (e) {
        const arrowKeys = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"];
        if (arrowKeys.includes(e.key)) {
            e.preventDefault();
        }
    },
    { passive: false }
);

// --- KEYDOWN Handler ---
window.addEventListener("keydown", function (e) {
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
            // Using 0.8 so that wheels “block” rotation instead of full stop
            a[2] = +0.8;
            break;
        default:
            // do nothing for other keys
            break;
    }
});

// --- KEYUP Handler ---
window.addEventListener("keyup", function (e) {
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
            // nothing to do
            break;
    }
});

// --------------
// 6. Kick off negotiation as soon as the page loads
// --------------
window.addEventListener("load", () => {
    negotiate().catch((err) => {
        console.error("negotiate() failed:", err);
    });
});
