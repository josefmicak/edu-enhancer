function loadQuestionInfo(itemNumberIdentifier, itemNameIdentifier, title, label, points, questionPointsDetermined) {
    document.getElementById("testtemplate-item-itemnumberidentifier").innerHTML = itemNumberIdentifier;
    document.getElementById("testtemplate-item-itemnameidentifier").innerHTML = itemNameIdentifier;
    document.getElementById("testtemplate-item-title").innerHTML = title;
    document.getElementById("testtemplate-item-label").innerHTML = label;

    if (questionPointsDetermined == "False") {
        points = "N/A";
    }
    document.getElementById("testtemplate-item-points").innerHTML = points;
}

function updatePointsInputs(correctAnswerCount, questionTypeText) {
    if (questionTypeText === "Seřazení pojmů") {
        correctAnswerCount = 1;
    }
    const subquestionPoints = document.getElementById("subquestion-points").value;
    const subquestionPointsPerAnswer = subquestionPoints / (correctAnswerCount <= 0 ? 1 : correctAnswerCount);
    document.getElementById("subquestion-correct-choice-points").value = Math.round(subquestionPointsPerAnswer * 100) / 100;
    document.getElementById("wrongChoicePoints_recommended_points").value = (Math.round(subquestionPointsPerAnswer * 100) / 100) * (-1);
    document.getElementById("wrongChoicePoints_selected_points").min = subquestionPoints * (-1);
}

function limitDecimalPlaces(el, decimalPlacesCount) {
    if (el.value) {
        if (decimalPlacesCount <= 0) { el.value = Math.round(el.value); }
        else { el.value = Math.round(parseFloat(el.value) * Math.pow(10, decimalPlacesCount)) / Math.pow(10, decimalPlacesCount); }
    }
}

function loadSolvedTestDetails(studentName, studentLogin, studentEmail) {
    document.getElementById("managesolvedtestlist-student-name").innerHTML = studentName;
    document.getElementById("managesolvedtestlist-student-login").innerHTML = studentLogin;
    document.getElementById("managesolvedtestlist-student-email").innerHTML = studentEmail;
}

function updateUserInputsWithRole(value) {
    if(document.getElementById("studentNumberIdentifier")) {
        if(value > 0 || document.getElementById("studentNumberIdentifier").querySelectorAll("option").length <= 1) {
            document.getElementById("studentNumberIdentifier").disabled = true;
        }
        else{
            document.getElementById("studentNumberIdentifier").disabled = false;
        }
    }
}

function updateUserInputsWithStudentNumberIdentifier(value) {
    if(document.getElementById("studentNumberIdentifier-value")) {
        document.getElementById("studentNumberIdentifier-value").value = value;
    }
}

// Dark theme
function handlePreferedColorSchemeChange(e) {
    if(e.matches) {
        if(document.getElementsByClassName("g_id_signin")[0]) {
            document.getElementsByClassName("g_id_signin")[0].dataset.theme = "filled_blue";
        }
    }
    else {
        if(document.getElementsByClassName("g_id_signin")[0]) {
            document.getElementsByClassName("g_id_signin")[0].dataset.theme = "outline";
        }
    }
}

const preferedColorScheme = window.matchMedia("(prefers-color-scheme: dark)");
handlePreferedColorSchemeChange(preferedColorScheme);

/*console.log(document.cookie);
let auth2;
let googleUser;

function handleCredentialResponse(response) {
    const responsePayload = decodeJwtResponse(response.credential);

    console.log("ID: " + responsePayload.sub);
    console.log('Full Name: ' + responsePayload.name);
    console.log('Given Name: ' + responsePayload.given_name);
    console.log('Family Name: ' + responsePayload.family_name);
    console.log("Image URL: " + responsePayload.picture);
    console.log("Email: " + responsePayload.email);
}

function decodeJwtResponse(token) {
    var base64Url = token.split('.')[1];
    var base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    var jsonPayload = decodeURIComponent(atob(base64).split('').map(function (c) {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
    }).join(''));

    return JSON.parse(jsonPayload);
};

function signOut() {
    console.log("Sign out");
    google.accounts.id.disableAutoSelect();
}*/