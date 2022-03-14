// Please see documentation at https://docs.microsoft.com/aspnet/core/client-side/bundling-and-minification
// for details on configuring this project to bundle and minify static web assets.

// Write your JavaScript code.

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

function updatePointsInputs(correctAnswerCount) {
    const subquestionPoints = document.getElementById("subquestion-points").value;
    const subquestionPointsPerAnswer = subquestionPoints / (correctAnswerCount <= 0 ? 1 : correctAnswerCount);
    document.getElementById("subquestion-correct-choice-points").value = subquestionPointsPerAnswer;
    document.getElementById("wrongChoicePoints_recommended_points").value = subquestionPointsPerAnswer * (-1);
}

function loadSolvedTestDetails(studentName, studentLogin, studentEmail) {
    document.getElementById("managesolvedtestlist-student-name").innerHTML = studentName;
    document.getElementById("managesolvedtestlist-student-login").innerHTML = studentLogin;
    document.getElementById("managesolvedtestlist-student-email").innerHTML = studentEmail;
}