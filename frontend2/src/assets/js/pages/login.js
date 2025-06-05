// --------------
// 1. Configuration
// --------------
const backendHost = "172.24.0.83:443"; // your backend address (with port)
const REST_URL = `https://${backendHost}/token`;

// Wait for the entire DOM to be loaded before running any code
document.addEventListener("DOMContentLoaded", () => {
    // -------------- 
    // get references to the login form the button sign in
    // --------------
    const loginButton = document.getElementById("loginButton");

    // add a click event listener to the login button
    loginButton.addEventListener("click", async function () {
        //Get the info the values from the email and password fields
        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;

        // Prepare form data as URL
        const formData = new URLSearchParams();
        formData.append("username", email);
        formData.append("password", password);

        window.location.href = 'dashboard/index.html';

        // Send a POST request to the backend to get the token
        
        try{
            // Send a post request to the FastAPI backend
            const response = await fetch(REST_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: formData,
            });

            //If the response is not ok, throw an error and create an alert

                  if (response.ok) {
                    // Parse the JSON response to extract the access token
                    const data = await response.json();
                    
                    // Save the access token in localStorage for future requests
                    localStorage.setItem('token', data.access_token);
                    
                    // Redirect the user to the dashboard page
                    window.location.href = 'dashboard/index.html';
                } else {
                    // If login fails (e.g., 403), show the error popup
                    showErrorPopup();
                }
                } catch (error) {
                // In case of a network error or other unexpected error, also show the popup
                showErrorPopup();
                }
            });
            });

        // Function to display an error popup for 5 seconds
        function showErrorPopup() {
        // Get the popup element by its ID
        const popup = document.getElementById('errorPopup');
        // Make it visible
        popup.style.display = 'block';
        // After 5 seconds, hide it again
        setTimeout(() => {
            popup.style.display = 'none';
        }, 5000);
        
    }




