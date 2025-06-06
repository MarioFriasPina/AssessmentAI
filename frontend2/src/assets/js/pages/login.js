// --------------
// 1. Configuration
// --------------
const backendHost = "10.49.12.47:9999/api"; // your backend address (with port)
const REST_URL = `https://${backendHost}/token`;


// Wait for the entire DOM to be loaded before running any code
document.addEventListener("DOMContentLoaded", () => {
    // --------------
    // Get reference to the login button
    // --------------
    const loginButton = document.getElementById("loginButton");

    // Add a click event listener to the login button
    loginButton.addEventListener("click", async function () {
        // Get the values from the email and password fields
        const email = document.getElementById("email").value.trim();
        const password = document.getElementById("password").value.trim();

        // Validate fields are not empty
        if (!email || !password) {
            showErrorPopup("Please fill in all fields.");
            return;
        }

        

        try {
            // Prepare form data as URL-encoded
            const formData = new URLSearchParams();
            formData.append("username", email);
            formData.append("password", password);
            // Send a POST request to the FastAPI backend
            const response = await fetch(REST_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: formData,
            });

            if (response.ok) {
                // Parse the JSON response to extract the access token
                const data = await response.json();

                // Save the access token in localStorage for future requests
                localStorage.setItem('token', data.access_token);

                // Redirect the user to the dashboard page
                window.location.href = 'dashboard/index.html';

                data = parseJwt(data.access_token);
                user = data.user;

                storedName = localStorage.setItem('name', user);

               

                
            } else {
                // If login fails (e.g., 403), show the error popup
                showErrorPopup("Incorrect credentials. Please try again.");
            }
        } catch (error) {
            // In case of a network error or other unexpected error, show the popup
            showErrorPopup("Error connecting to the server.");
        }
    });
});

// Function to display an error popup for 5 seconds
function showErrorPopup(message = "Invalid email or password..") {
    const popup = document.getElementById('errorPopup');
    popup.textContent = message; // Set the message
    popup.style.display = 'block'; // Make it visible

    // After 5 seconds, hide it again
    setTimeout(() => {
        popup.style.display = 'none';
    }, 5000);
}

function parseJwt (token) {
    // Decode the JWT token and return the payload as a JSON object
    console.log("Parsing JWT token:");
    var base64Url = token.split('.')[1];
    var base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    var jsonPayload = decodeURIComponent(window.atob(base64).split('').map(function(c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));

    return JSON.parse(jsonPayload);
}
