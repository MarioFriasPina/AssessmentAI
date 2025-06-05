document.addEventListener("DOMContentLoaded", () => {
  const tableBody = document.getElementById("scoreTableBody");

  const scoreboardData = [
    { name: "John Doe", time: "00:45", date: "2025-06-01" },
    { name: "Jane Smith", time: "01:12", date: "2025-06-02" },
    { name: "Alice Brown", time: "00:59", date: "2025-06-03" },
    { name: "Bob Johnson", time: "01:23", date: "2025-06-03" },
    { name: "Alan Hernandez", time: "00:39", date: "2025-06-04" },
  ];

  scoreboardData.forEach(entry => {
    const row = document.createElement("tr");

    row.innerHTML = `
      <td><a href="#" class="text-muted">${entry.name}</a></td>
      <td>${entry.time}</td>
      <td>${entry.date}</td>
    `;

    tableBody.appendChild(row);
  });
});

// Function to display metrics in the modal

document.addEventListener("DOMContentLoaded", () => {
  const stats = {
    bestTime: "12 minutes",
    avgTime: "5 minutes",
    wins: "17 wins",
    losses: "8 losses"
  };

  //check if the data is null or undefined
  //if null or undefined, we will display 0
    if (!stats.bestTime) stats.bestTime = "0 minutes";
    if (!stats.avgTime) stats.avgTime = "0 minutes";
    if (!stats.wins) stats.wins = "0 wins";
    if (!stats.losses) stats.losses = "0 losses";


  // Insertamos los valores en el HTML
  document.getElementById("bestTime").textContent = stats.bestTime;
  document.getElementById("avgTime").textContent = stats.avgTime;
  document.getElementById("wins").textContent = stats.wins;
  document.getElementById("losses").textContent = stats.losses;
  
});

