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
