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

// Default theme
let theme = sessionStorage.getItem("theme");

// Prefered color scheme
function handlePreferedColorSchemeChange(e) {
    if (e.matches) {
        setTheme("dark");
        
    }
    else {
        setTheme("light");
    }
}

const preferedColorScheme = window.matchMedia("(prefers-color-scheme: dark)");
if (theme == null) {
    preferedColorScheme.addEventListener("change", (e) => {
        handlePreferedColorSchemeChange(preferedColorScheme);
    });
    handlePreferedColorSchemeChange(preferedColorScheme);
}
else {
    setTheme(theme);
}

// Change theme
function changeTheme(el) {
    if (theme == "dark") {
        setTheme("light", true);
    }
    else {
        setTheme("dark", true);
    }
}

// Set theme
function setTheme(toTheme, saveTheme=false) {
    theme = toTheme;
    if (saveTheme) { sessionStorage.setItem("theme", toTheme); }

    if (toTheme == "dark") {
        if (document.documentElement) {
            document.documentElement.classList.add("dark-theme");
        }
        if (document.getElementsByClassName("g_id_signin")[0]) {
            document.getElementsByClassName("g_id_signin")[0].dataset.theme = "filled_blue";
        }
        if (document.getElementById("themeBtn")) {
            document.getElementById("themeBtn").innerHTML = "light_mode";
            document.getElementById("themeBtn").title = "Přepnout do světlého režimu";
        }
    }
    else {
        if (document.documentElement) {
            document.documentElement.classList.remove("dark-theme");
        }
        if (document.getElementsByClassName("g_id_signin")[0]) {
            document.getElementsByClassName("g_id_signin")[0].dataset.theme = "outline";
        }
        if (document.getElementById("themeBtn")) {
            document.getElementById("themeBtn").innerHTML = "dark_mode";
            document.getElementById("themeBtn").title = "Přepnout do tmavého režimu";
        }
    }
}