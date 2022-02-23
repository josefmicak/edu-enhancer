// Please see documentation at https://docs.microsoft.com/aspnet/core/client-side/bundling-and-minification
// for details on configuring this project to bundle and minify static web assets.

// Write your JavaScript code.

function loadQuestionInfo(itemNumberIdentifier, itemNameIdentifier, title, label, points, questionPointsDetermined) {
    document.getElementById("testtemplate-item-itemnumberidentifier").innerHTML = "Číselný identifikátor otázky: " + itemNumberIdentifier;
    document.getElementById("testtemplate-item-itemnameidentifier").innerHTML = "Jmenný identifikátor otázky: " + itemNameIdentifier;
    document.getElementById("testtemplate-item-title").innerHTML = "Nadpis otázky: " + title;
    document.getElementById("testtemplate-item-label").innerHTML = "Označení otázky: " + label;

    if (questionPointsDetermined == "False") {
        points = "N/A";
    }
    document.getElementById("testtemplate-item-points").innerHTML = "Počet bodů za otázku: " + points;
}

function disableSelectedWrongChoicePointsTextbox() {
    document.getElementById("wrongChoicePoints_selected_points").disabled = true
}

function loadSolvedTestDetails(studentName, studentLogin, studentEmail) {
    document.getElementById("managesolvedtestlist-student-name").innerHTML = "Jméno studenta: " + studentName;
    document.getElementById("managesolvedtestlist-student-login").innerHTML = "Login studenta: " + studentLogin;
    document.getElementById("managesolvedtestlist-student-email").innerHTML = "Email studenta: " + studentEmail;
}