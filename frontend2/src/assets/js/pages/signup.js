// --------------
// 1. Configuration
// --------------
const backendHost = "172.24.0.83:443"; // your backend address (with port)
const REST_URL = `https://${backendHost}/create`;

// Wait for the entire DOM to be loaded before running any code
document.addEventListener("DOMContentLoaded", () => {
    // -------------- 
    // get references to the login form the button sign in
    // --------------
    const signUp = document.getElementById("signUpButton");

    // add a click event listener to the login button
    signUp.addEventListener("click", async function () {
        //Get the info the values from the email and password fields
        const name = document.getElementById("name").value;
        const last_name = document.getElementById("last_name").value;
        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;


        // Prepare form data as URL
        const formData = new URLSearchParams();
        formData.append("name", name);
        formData.append("name_last", last_name);
        formData.append("email", email);
        formData.append("password", password);

        window.location.href = '/login.html';

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

                  if (response = "success") {
                    // redirect the user to the login page
                    window.location.href = 'login.html';
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



