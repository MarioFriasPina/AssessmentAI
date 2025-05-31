const backendHost = "localhost:8080";

async function negotiate() {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const resp = await fetch(`http://${backendHost}/offer`, {
        method: "POST",
        body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type,
            session_id: session_id,
        }),
        headers: { "Content-Type": "application/json" },
    });
    const ans = await resp.json();
    await pc.setRemoteDescription(ans);
}

function uuidv4() {
    return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, (c) =>
        (
            c ^
            (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))
        ).toString(16)
    );
}

const session_id = uuidv4();

document.cookie = `session_id=${session_id}; path=/; max-age=${
    60 * 60 * 24 * 7
}`;

const video_user = document.getElementById("video_user");
const video_rl = document.getElementById("video_rl");
const loading_user = document.getElementById("loading_user");
const loading_rl = document.getElementById("loading_rl");
let pc = new RTCPeerConnection();

pc.addTransceiver("video", { direction: "recvonly" });
pc.addTransceiver("video", { direction: "recvonly" });

pc.ontrack = (evt) => {
    if (evt.track.kind === "video") {
        if (!video_user.srcObject) {
            video_user.srcObject = new MediaStream([evt.track]);
            loading_user.style.display = "none";
        } else if (!video_rl.srcObject) {
            video_rl.srcObject = new MediaStream([evt.track]);
            loading_rl.style.display = "none";
        }
    }
};

async function negotiate() {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const resp = await fetch("/offer", {
        method: "POST",
        body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type,
            session_id: session_id,
        }),
        headers: { "Content-Type": "application/json" },
    });
    const ans = await resp.json();
    await pc.setRemoteDescription(ans);
}

negotiate();

// Connect to WebSocket with session_id as query param
const ws = new WebSocket(
    "ws://" + backendHost + "/ws?session_id=" + session_id
);

function getAction(key) {
    if (key === "ArrowUp") return 3;
    if (key === "ArrowDown") return 4;
    if (key === "ArrowLeft") return 1;
    if (key === "ArrowRight") return 2;
    return 0;
}

document.addEventListener("keydown", (e) => {
    const action = getAction(e.key);
    ws.send(JSON.stringify({ action }));
});

document.addEventListener("keyup", (e) => {
    // Send no-op when key is released
    ws.send(JSON.stringify({ action: 0 }));
});
