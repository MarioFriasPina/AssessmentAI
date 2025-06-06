const backendHost = "10.49.12.47:9999/api"; // your backend address (with port)
const REST_URL = `https://${backendHost}/leaderboard`;
const REST_URL2 = `https://${backendHost}/data`;

const token = localStorage.getItem('token');

document.addEventListener("DOMContentLoaded", async () => {
  const tableBody = document.getElementById("scoreTableBody");

  if (!token) {
    console.error("No token found in localStorage.");
    return;
  }

  try {
    const response = await fetch(REST_URL, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const scoreboardData = await response.json();

    scoreboardData.forEach(entry => {
      const row = document.createElement("tr");

      row.innerHTML = `
        <td><a href="#" class="text-muted">${entry.name}</a></td>
        <td>${entry.score}</td>
        <td>${new Date(entry.date).toLocaleString()}</td>
      `;

      tableBody.appendChild(row);
    });

  } catch (error) {
    console.error("Error fetching leaderboard:", error);
  }
});

// Function to display metrics in the modal

document.addEventListener("DOMContentLoaded", async () => {
  if (!token) {
    console.error("No token found in localStorage.");
    return;
  }

  try {
    const response = await fetch(REST_URL2, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const stats = await response.json();

    // Fallbacks por si viene null o undefined
    const bestTime = stats.record ? `${stats.record} minutes` : "0 minutes";
    const avgTime = stats.total ? `${Math.round(stats.record / stats.total)} minutes` : "0 minutes";
    const wins = stats.win ?? 0;
    const losses = stats.loss ?? 0;

    // Insertamos los valores en el HTML
    document.getElementById("bestTime").textContent = bestTime;
    document.getElementById("avgTime").textContent = avgTime;
    document.getElementById("wins").textContent = `${wins} wins`;
    document.getElementById("losses").textContent = `${losses} losses`;

  } catch (error) {
    console.error("Error fetching user stats:", error);
  }
});