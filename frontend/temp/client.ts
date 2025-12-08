const ws = new WebSocket("ws://localhost:8000/ws/agent");
ws.onopen = () => {
    // optionally send a JSON question; otherwise server uses default
    ws.send(JSON.stringify({ question: "What is the Wi‑Fi password for guests?" }));
};
ws.onmessage = (evt) => {
    const obj = JSON.parse(evt.data);
    if (obj.type === "chunk") {
        // append partial text to UI
        console.log("chunk:", obj.text);
    } else if (obj.type === "done") {
        console.log("stream complete");
    } else if (obj.type === "error") {
        console.error("error:", obj.message);
    }
};
ws.onclose = () => console.log("closed");
