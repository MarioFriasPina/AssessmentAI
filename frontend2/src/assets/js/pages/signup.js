// --------------
// 1. Configuration
// --------------
const backendHost = "172.24.0.83:443/api"; // your backend address (with port)
const REST_URL = `https://${backendHost}/create`;

// Wait for the entire DOM to be loaded before running any code
document.addEventListener("DOMContentLoaded", () => {
    // Get reference to the sign-up button
    const signUp = document.getElementById("signUpButton");

    // Add a click event listener
    signUp.addEventListener("click", async function () {
        // Get the values from input fields
        const name = document.getElementById("name").value.trim();
        const last_name = document.getElementById("last_name").value.trim();
        const email = document.getElementById("email").value.trim();
        const password = document.getElementById("password").value.trim();

        

        // Validate fields are not empty
        if (!name || !last_name || !email || !password) {
            showErrorPopup("Please fill in all fields.");
            return;
        }

        else if (password.length < 8) {
            showErrorPopup("Password must be at least 8 characters long.");
            return;
        }

        // Prepare form data as URL-encoded
     
        try {

            var dataToSend = {
                "name": name,
                "name_last": last_name,
                "email": email,
                "password": password
            };


            // Send a POST request to the FastAPI backend
            const response = await fetch(REST_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body:  JSON.stringify(dataToSend)
            });

            if (response.ok) {
                const result = await response.json();
                showSuccessPopup("Registration successful! You can now log in.");
                setTimeout(() => {
                    window.location.href = "login.html"; // Redirect to login page after success
                }, 1000);
            } else {
                showErrorPopup("Registration failed. Please check your data.");
            }
        } catch (error) {
            showErrorPopup("Unable to connect to the server.");
        }
    });
});

// Function to display an error popup for 5 seconds
function showErrorPopup(message = "An error occurred.") {
    const popup = document.getElementById('errorPopup');
    popup.textContent = message;
    popup.style.display = 'block';
    popup.style.color = 'red';
    setTimeout(() => {
        popup.style.display = 'none';
    }, 400);
}

// Function to display a success popup in green
function showSuccessPopup(message = "Success.") {
    const popup = document.getElementById('errorPopup');
    popup.textContent = message;
    popup.style.display = 'block';
    popup.style.color = 'green';
    setTimeout(() => {
        popup.style.display = 'none';
    }, 400);
}
