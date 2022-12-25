﻿//ManageUserRegistrationList.cshtml

window.updateVisibility = function (accepted, rejected, text) {
    if (accepted == false) {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[8].textContent === "Schválena") {
                tr.style.display = 'none';
            }
        });
    }
    else {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[8].textContent === "Schválena") {
                tr.style.display = '';
            }
        });
    }

    if (rejected == false) {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[8].textContent === "Zamítnuta") {
                tr.style.display = 'none';
            }
        });
    }
    else {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[8].textContent === "Zamítnuta") {
                tr.style.display = '';
            }
        });
    }

    if (text == true) {
        document.getElementById('visibility-text').innerHTML = 'Změny úspěšně uloženy.';
    }
}

//ManageUserList.cshtml / ManageUserListForAdmin.cshtml

function addStudentDetails(clicked_id) {
    hideEditStudentLabel();
    const idArray = clicked_id.split("_");    
    var table = document.getElementById("student-table"); 

    var studentIdentifier = table.rows[idArray[1]].cells[0].innerHTML;
    document.getElementById("studentIdentifier").value = studentIdentifier;
    document.getElementById('studentIdentifier').readOnly = true;

    var fullName = table.rows[idArray[1]].cells[1].innerHTML;
    const nameArray = fullName.split(" "); 
    document.getElementById("studentFirstName").value = nameArray[0];
    document.getElementById('studentFirstName').readOnly = true;
    document.getElementById("studentLastName").value = nameArray[1];
    document.getElementById('studentLastName').readOnly = true;

    var login = table.rows[idArray[1]].cells[2].innerHTML;
    document.getElementById("studentLogin").value = login;
    document.getElementById('studentLogin').readOnly = true;
}

function showEditStudentLabel(oldLogin, userIdentifier, email, firstName, lastName) {
    document.getElementById("student-action").value = 'editStudent';
    document.getElementById("added-student").style.visibility = 'hidden';
    document.getElementById("edited-student").style.visibility = 'visible';
    document.getElementById("studentOldLogin").value = oldLogin;
    document.getElementById("studentIdentifier").value = userIdentifier;
    document.getElementById('studentIdentifier').readOnly = false;
    document.getElementById("studentFirstName").value = firstName;
    document.getElementById('studentFirstName').readOnly = false;
    document.getElementById("studentLastName").value = lastName;
    document.getElementById('studentLastName').readOnly = false;
    document.getElementById("studentLogin").value = oldLogin;
    document.getElementById('studentLogin').readOnly = true;
    document.getElementById("studentEmail").value = email;
}

function hideEditStudentLabel() {
    document.getElementById("student-action").value = 'addStudent';
    document.getElementById("added-student").style.visibility = 'visible';
    document.getElementById("edited-student").style.visibility = 'hidden';
    document.getElementById("studentIdentifier").value = "";
    document.getElementById("studentFirstName").value = "";
    document.getElementById("studentLastName").value = "";
    document.getElementById("studentLogin").value = "";
    document.getElementById('studentLogin').readOnly = false;
    document.getElementById("studentEmail").value = "";
}

function showEditTeacherLabel(oldLogin, email, firstName, lastName, makeVisible) {
    document.getElementById("teacher-action").value = 'editTeacher';
    document.getElementById("added-teacher").style.visibility = 'hidden';
    document.getElementById("edited-teacher").style.visibility = 'visible';
    document.getElementById("teacherOldLogin").value = oldLogin;
    document.getElementById("teacherFirstName").value = firstName;
    document.getElementById("teacherLastName").value = lastName;
    document.getElementById("teacherLogin").value = oldLogin;
    document.getElementById('teacherLogin').readOnly = true;
    document.getElementById("teacherEmail").value = email;

    if (makeVisible) {
        document.getElementById("teacher-edit-role").style.visibility = 'visible';
    }
}

function hideEditTeacherLabel() {
    document.getElementById("teacher-action").value = 'addTeacher';
    document.getElementById("added-teacher").style.visibility = 'visible';
    document.getElementById("edited-teacher").style.visibility = 'hidden';
    document.getElementById("teacher-edit-role").style.visibility = 'hidden';
    document.getElementById("teacherFirstName").value = "";
    document.getElementById("teacherLastName").value = "";
    document.getElementById("teacherLogin").value = "";
    document.getElementById('teacherLogin').readOnly = false;
    document.getElementById("teacherEmail").value = "";
}

function showEditAdminLabel(oldLogin, email, firstName, lastName, role) {
    document.getElementById("admin-action").value = 'editAdmin';
    document.getElementById("added-admin").style.visibility = 'hidden';
    document.getElementById("edited-admin").style.visibility = 'visible';
    document.getElementById("admin-edit-role").style.visibility = 'visible';
    document.getElementById("adminOldLogin").value = oldLogin;
    document.getElementById("adminFirstName").value = firstName;
    document.getElementById("adminLastName").value = lastName;
    document.getElementById("adminLogin").value = oldLogin;
    document.getElementById('adminLogin').readOnly = true;
    document.getElementById("adminEmail").value = email;
    
    if (role == "Admin") {
        document.getElementById("isMainAdmin").value = false;
    }
    else if (role == "MainAdmin") {
        document.getElementById("isMainAdmin").value = true;
    }
}

function hideEditAdminLabel() {
    document.getElementById("admin-action").value = 'addAdmin';
    document.getElementById("added-admin").style.visibility = 'visible';
    document.getElementById("edited-admin").style.visibility = 'hidden';
    document.getElementById("admin-edit-role").style.visibility = 'hidden';
    document.getElementById("adminFirstName").value = "";
    document.getElementById("adminLastName").value = "";
    document.getElementById("adminLogin").value = "";
    document.getElementById('adminLogin').readOnly = false;
    document.getElementById("adminEmail").value = "";
}

function changeAdminRole(checkbox) {
    const checked = checkbox.checked;
    if (checked) {
        document.getElementById("change-admin-role").disabled = false;
    }
    else {
        document.getElementById("change-admin-role").disabled = true;
    }
}

function adminFormSubmit(event) {
    var isMainAdmin = document.getElementById("isMainAdmin").value;
    var role = document.getElementById("change-admin-role").value;

    if (isMainAdmin == "false" && role == "4") {
        document.getElementById("email").value = document.getElementById("adminEmail").value;
        document.getElementById("login").value = document.getElementById("adminLogin").value;
        document.getElementById("firstName").value = document.getElementById("adminFirstName").value;
        document.getElementById("lastName").value = document.getElementById("adminLastName").value;
        document.getElementById("role").value = role;
        document.getElementById("action").value = 'changeMainAdmin';

        showConfirmActionForm("changeMainAdmin", null, null, null, null, null, null);
        event.preventDefault();
    }
}

//check if admin-form exists to prevent invalid JS element call
var elementExists = document.getElementById("admin-form");
if (elementExists) {
    const adminForm = document.getElementById('admin-form');
    adminForm.addEventListener('submit', adminFormSubmit);
}

//QuestionTemplate.cshtml

function setWrongChoicePointsInputs(el) {
    if (el.value == "wrongChoicePoints_automatic_radio") {
        document.getElementById("wrongChoicePoints_manual").readOnly = true;
    }
    else {
        document.getElementById("wrongChoicePoints_manual").readOnly = false;
    }
}

//show form which prompts user to confirm the action

function showConfirmActionForm(action, identifier, email, login, firstName, lastName, role) {
    document.getElementById("confirm-action").style.display = "block";
    document.getElementById("action").value = action;
    if (action == "deleteTemplate") {
        document.getElementById("testNumberIdentifier").value = identifier;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit testovou šablonu s identifikátorem '" + identifier + "'?";
    }
    else if (action == "deleteAllTemplates") {
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit všechny testovací šablony?";
    }
    else if (action == "deleteQuestionTemplate") {
        document.getElementById("questionNumberIdentifier").value = identifier;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit zadání otázky s identifikátorem '" + identifier + "'?";
    }
    else if (action == "deleteSubquestionTemplate") {
        document.getElementById("subquestionIdentifierToDelete").value = identifier;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit zadání podotázky s identifikátorem '" + identifier + "'?";
    }
    else if (action == "deleteResult") {
        document.getElementById("testResultIdentifier").value = identifier;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit výsledek s identifikátorem '" + identifier + "'?";
    }
    else if (action == "deleteAllResults") {
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit všechny výsledky testů?";
    }
    else if (action == "deleteRegistration" || action == "refuseRegistration") {
        document.getElementById("email").value = email;
        if (action == "deleteRegistration") {
            document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete smazat registraci s emailem '" + email + "'?";
        }
        else if (action == "refuseRegistration") {
            document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete zamítnout registraci s emailem '" + email + "'?";
        }
    }
    else if (action == "acceptRegistration") {
        document.getElementById("email").value = email;
        document.getElementById("login").value = login;
        document.getElementById("firstName").value = firstName;
        document.getElementById("lastName").value = lastName;
        document.getElementById("role").value = role;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete přijmout registraci s emailem '" + email + "'?";
    }
    else if (action == "deleteStudent" || action == "deleteTeacher" || action == "deleteAdmin") {
        document.getElementById("login").value = login;
        if (action == "deleteStudent") {
            document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete smazat studenta s loginem '" + login + "'?";
        }
        else if (action == "deleteTeacher") {
            document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete smazat učitele s loginem '" + login + "'?";
        }
        else if (action == "deleteAdmin") {
            document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete smazat správce s loginem '" + login + "'?";
        }
    }
    else if (action == "deleteAllStudents") {
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit všechny studenty?";
    }
    else if (action == "deleteAllTeachers") {
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit všechny učitele?";
    }
    else if (action == "deleteAllAdmins") {
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit všechny správce?";
    }
    else if (action == "changeMainAdmin") {
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete změnit tohoto správce na hlavního administrátora?" + 
            " (provedením této akce bude váš účet změněn z hlavního administrátora na správce).";
    }
}

//AddSubquestionTemplate.cshtml


function getOptionPlaceholderText() {
    return '-ZVOLTE MOŽNOST-';
}

function addPossibleAnswer() {
    var table = document.getElementById('possible-answers-table');
    var rowCount = table.rows.length;
    var lastRowInnerHTML = table.rows[rowCount - 1].innerHTML;
    var lastRowIdArray = table.rows[rowCount - 1].id.split("-");
    var lastRowId = parseInt(lastRowIdArray[2]);
    lastRowId += 1;
    var row = table.insertRow(rowCount);
    row.innerHTML = lastRowInnerHTML;
    row.id = "possible-answer-" + lastRowId;
}

function editPossibleAnswers(action, subquestionType) {
    //users wants to edit possible answers
    if (action == "enable") {
        if (subquestionType == 10) {
            $(".slider-input").prop('disabled', false);
            document.getElementById("subquestion-add").disabled = true;
            document.getElementById("possible-answer-save").disabled = false;
            document.getElementById("correct-answer-edit").disabled = true;
            document.getElementById("possible-answer-edit").disabled = true;
        }
        else {
            document.getElementById("possible-answer-add").disabled = false;
            document.getElementById("possible-answer-edit").disabled = true;
            document.getElementById("correct-answer-edit").disabled = true;
            document.getElementById("subquestion-add").disabled = true;
            $(".possible-answer-delete").prop('disabled', false);
            document.getElementById("possible-answer-save").disabled = false;
            $(".possible-answer-input").prop('readonly', false);
            $(".possible-answer-move").prop('disabled', false);
        }
    }
    //user is done editing possible answers
    else if (action == "disable") {
        var addAnswers = true;
        var seen = {};
        var min = 0;
        var max = 0;

        $('input[type="text"].possible-answer-input').each(function () {
            var answer = $(this).val();
            if (answer.length == 0) {
                alert("Chyba: nevyplněná možná odpověď.");
                addAnswers = false;
                return false;
            }
            if (seen[answer]) {
                alert("Chyba: duplikátní možná odpověď (" + answer + ").");
                addAnswers = false;
                return false;
            }
            else {
                seen[answer] = true;
            }  
        });

        if (subquestionType == 10) {
            min = document.getElementById("slider-min").value;
            max = document.getElementById("slider-max").value;
            if (min.length == 0 || max.length == 0) {
                alert("Chyba: nevyplněná možná odpověď.");
                addAnswers = false;
            }
            if (min >= max) {
                alert("Chyba: maximální hodnota musí být vyšší než minimální hodnota.");
                addAnswers = false;
            }
        }

        if (addAnswers) {
            if (subquestionType == 1) {
                updateCorrectAnswersInput();
                document.getElementById("subquestion-add").disabled = false;

                document.getElementById("possible-answer-add").disabled = true;
                $(".possible-answer-delete").prop('disabled', true);
                document.getElementById("possible-answer-save").disabled = true;
                document.getElementById("possible-answer-edit").disabled = false;
                document.getElementById("correct-answer-edit").disabled = false;
                $(".possible-answer-input").prop('readonly', true);
                $(".possible-answer-move").prop('disabled', true);
                var subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
            }
            else if (subquestionType == 2 || subquestionType == 3 || subquestionType == 6 || subquestionType == 7) {
                document.getElementById("subquestion-add").disabled = true;
                updateCorrectAnswersSelect("possibleAnswersModified", subquestionType);

                document.getElementById("possible-answer-add").disabled = true;
                $(".possible-answer-delete").prop('disabled', true);
                document.getElementById("possible-answer-save").disabled = true;
                document.getElementById("possible-answer-edit").disabled = false;
                document.getElementById("correct-answer-edit").disabled = false;
                $(".possible-answer-input").prop('readonly', true);
                $(".possible-answer-move").prop('disabled', true);
                var subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
            }
            else if (subquestionType == 10) {
                $(".slider-input").prop('disabled', true);
                document.getElementById("slider-question").min = min;
                document.getElementById("slider-question").max = max;
                var sliderQuestion = document.getElementById("slider-question");
                sliderQuestion.value = Math.round((parseInt(min) + parseInt(max)) / 2);
                sliderQuestion.nextElementSibling.value = Math.round((parseInt(min) + parseInt(max)) / 2);
                document.getElementById("possible-answer-save").disabled = true;
                document.getElementById("possible-answer-edit").disabled = false;
                document.getElementById("correct-answer-edit").disabled = false;

                var subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
            }
        }
    }
}

function deletePossibleAnswer(clicked_id, subquestionType) {
    var minPossibleAnswers = [0, 2, 2, 4, 0, 0, 2, 2, 0, 2, 0];
    var table = document.getElementById('possible-answers-table');
    var rowCount = table.rows.length;
    if (rowCount <= minPossibleAnswers[subquestionType] + 1) {
        alert('Chyba: musí existovat alespoň ' + minPossibleAnswers[subquestionType] + ' možné odpovědi.');
    }
    else {
        var row = document.getElementById(clicked_id);
        row.parentNode.removeChild(row);
    }
}

//enables the user to move correct answers (upwards or downwards)
function movePossibleAnswer(direction, clicked_id) {
    var table = document.getElementById('possible-answers-table');
    var rowIndex = 0;
    for (var i = 0, row; row = table.rows[i]; i++) {
        if (clicked_id == row.id) {
            rowIndex = i;
            break;
        }
    }
    var rowCount = table.rows.length;
    if (direction == 'up') {
        if (rowIndex > 1) {
            var temp = table.rows[rowIndex - 1].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex - 1].cells[0].getElementsByTagName("input")[0].value = table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value = temp;
        }
    }
    else if (direction == 'down') {
        if (rowIndex + 1 < rowCount) {
            var temp = table.rows[rowIndex + 1].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex + 1].cells[0].getElementsByTagName("input")[0].value = table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value = temp;
        }
    }
}

function addCorrectAnswer(subquestionType, isProgramatical) {
    var addAnswer = true;
    var correctAnswersTable = document.getElementById('correct-answers-table');

    if (subquestionType == 2) {
        //check if new correct answer can be added
        var possibleAnswersTable = document.getElementById('possible-answers-table');
        var possibleAnswersTableRowCount = possibleAnswersTable.rows.length;
        var correctAnswersTableRowCount = correctAnswersTable.rows.length;

        if (correctAnswersTableRowCount >= possibleAnswersTableRowCount) {
            addAnswer = false;
            alert('Chyba: může existovat maximálně ' + (possibleAnswersTableRowCount - 1) + " možných odpovědí.");
        }
    }

    if (addAnswer) {
        var rowCount = correctAnswersTable.rows.length;
        var lastRowInnerHTML = correctAnswersTable.rows[rowCount - 1].innerHTML;
        var lastRowIdArray = correctAnswersTable.rows[rowCount - 1].id.split("-");
        var lastRowId = parseInt(lastRowIdArray[2]);
        lastRowId += 1;

        if (subquestionType == 1) {//programmer use only
            var row = correctAnswersTable.insertRow(rowCount);
            row.innerHTML = lastRowInnerHTML;
            row.id = "correct-answer-" + lastRowId;
        }
        else if (subquestionType == 2 || subquestionType == 3) {
            var row = correctAnswersTable.insertRow(rowCount);
            row.innerHTML = lastRowInnerHTML;
            row.id = "correct-answer-" + lastRowId;

            $('.correct-answer-select').prop('disabled', false);
            //replace currently selected option with placeholder option
            if (!isProgramatical) {
                var correctAnswerSelects = document.getElementsByClassName('correct-answer-select');
                correctAnswerSelects[correctAnswerSelects.length - 1].options[0].innerHTML = getOptionPlaceholderText();
            }
        }
        else if (subquestionType == 4) {
            var lastRowRadioNameArray = correctAnswersTable.rows[rowCount - 1].cells[1].getElementsByTagName("input")[0].name.split("-");
            var lastRowRadioNumber = parseInt(lastRowRadioNameArray[3]);

            var yesChecked = true;//incicates whether the "yes" option is checked on the last row
            if (correctAnswersTable.rows[rowCount - 1].cells[2].getElementsByTagName("input")[0].checked) {
                yesChecked = false;
            }

            var row = correctAnswersTable.insertRow(rowCount);
            row.innerHTML = lastRowInnerHTML;
            row.id = "correct-answer-" + lastRowId;
            row.cells[1].getElementsByTagName("input")[0].name = "correct-answer-radio-" + parseInt(lastRowRadioNumber + 1);
            row.cells[2].getElementsByTagName("input")[0].name = "correct-answer-radio-" + parseInt(lastRowRadioNumber + 1);

            //after the new row is added, we check the previously checked radio button on the row that has been copied
            if (yesChecked) {
                correctAnswersTable.rows[rowCount - 1].cells[1].getElementsByTagName("input")[0].checked = true;
            }
            else {
                correctAnswersTable.rows[rowCount - 1].cells[2].getElementsByTagName("input")[0].checked = true;
            }
        }
    }
}

//after the user updates possible answers, correct answers must be automatically updated as well
//subquestion types - 1
function updateCorrectAnswersInput() {
    var possibleAnswerArray = [];
    possibleAnswerArray.push(getOptionPlaceholderText());
    $('input[type="text"].possible-answer-input').each(function () {
        var answer = $(this).val();
        possibleAnswerArray.push(answer);
    });

    //clear correct answers table
    var table = document.getElementById('correct-answers-table');
    var rowCount = table.rows.length;
    while (--rowCount - 1) {
        table.deleteRow(rowCount);
    } 
    table.rows[1].cells[0].getElementsByTagName("input")[0].value = "";

    for (var i = 0; i < possibleAnswerArray.length - 2; i++) {//todo: -1 ?
        rowCount = table.rows.length;
        var lastRowInnerHTML = table.rows[rowCount - 1].innerHTML;
        var lastRowIdArray = table.rows[rowCount - 1].id.split("-");
        var lastRowId = parseInt(lastRowIdArray[2]);
        lastRowId += 1;
        var row = table.insertRow(rowCount);
        row.innerHTML = lastRowInnerHTML;
        row.id = "correct-answer-" + lastRowId;
    }

    for (var i = 1; i < possibleAnswerArray.length; i++) {
        table.rows[i].cells[0].getElementsByTagName("input")[0].value = possibleAnswerArray[i];
    }
}

//after the user updates subquestion text, an appropriate number of correct ansers is added to the correct answers table
//subquestion types - 9
function updateCorrectAnswersInputFreeAnswer() {
    var additionalQuestions = document.getElementById("additional-questions");
    var table = document.getElementById('correct-answers-table');
    var rowCount = table.rows.length;
    while (--rowCount - 1) {
        table.deleteRow(rowCount);
    }
    table.rows[1].cells[0].getElementsByTagName("input")[0].value = "";
    table.rows[1].cells[0].getElementsByTagName("input")[0].placeholder = "[1] - Správná odpověď";

    var gapTexts = additionalQuestions.getElementsByClassName("gap-text");
    for (var i = 0; i < gapTexts.length; i++) {
        rowCount = table.rows.length;
        var lastRowInnerHTML = table.rows[rowCount - 1].innerHTML;
        var lastRowIdArray = table.rows[rowCount - 1].id.split("-");
        var lastRowId = parseInt(lastRowIdArray[2]);
        lastRowId += 1;
        var row = table.insertRow(rowCount);
        row.innerHTML = lastRowInnerHTML;
        row.id = "correct-answer-" + lastRowId;
        row.cells[0].getElementsByTagName("input")[0].placeholder = "[" + parseInt(i + 2) + "] - Správná odpověď";
    }
}

//automatic update of correct answers when dropdown menus are used after possible answers are modified
//subquestion types - 2, 3, 6
function updateCorrectAnswersSelect(performedAction, subquestionType) {
    //user modified possible answers - all correct answers are deleted and replaced by new possible answers
    if (performedAction == "possibleAnswersModified") {
        var possibleAnswerArray = [];
        possibleAnswerArray.push(getOptionPlaceholderText());
        $('input[type="text"].possible-answer-input').each(function () {
            var answer = $(this).val();
            possibleAnswerArray.push(answer);
        });

        //clear correct answers table
        var table = document.getElementById('correct-answers-table');
        var rowCount = table.rows.length;
        while (--rowCount - 1) {
            table.deleteRow(rowCount);
        } 
        $(".correct-answer-select").empty();

        //for some subquestion types, new rows must be added to the correct answer table
        if (subquestionType == 3) {
            //var table = document.getElementById('correct-answers-table');
            for (var i = 0; i < Math.floor((possibleAnswerArray.length - 3) / 2); i++) {//-3 because of 1 already existing row and 1 possible answer containing placeholder text
                rowCount = table.rows.length;
                var lastRowInnerHTML = table.rows[rowCount - 1].innerHTML;
                var lastRowIdArray = table.rows[rowCount - 1].id.split("-");
                var lastRowId = parseInt(lastRowIdArray[2]);
                lastRowId += 1;
                var row = table.insertRow(rowCount);
                row.innerHTML = lastRowInnerHTML;
                row.id = "correct-answer-" + lastRowId;
            }
        }

        var correctAnswerSelect = document.getElementsByClassName('correct-answer-select');
        for (var i = 0; i < correctAnswerSelect.length; i++) {
            for (var j = 0; j < possibleAnswerArray.length; j++) {
                var opt = document.createElement('option');
                opt.value = possibleAnswerArray[j];
                opt.innerHTML = possibleAnswerArray[j];
                correctAnswerSelect.item(i).appendChild(opt);
            }
        }

        var correctAnswerArray = [];
        $('select.correct-answer-select').each(function () {
            var answer = $(this).val();
            correctAnswerArray.push(answer);
        });
    }
    //user selected or deleted a correct answer
    //this selected or deleted answer is removed from all other dropdowns, while previously selected answer is added to all other dropdowns
    else if (performedAction == "correctAnswerChosen") {
        var possibleAnswerArray = [];
        $('input[type="text"].possible-answer-input').each(function () {
            var answer = $(this).val();
            possibleAnswerArray.push(answer);
        });

        var correctAnswerArray = [];
        $('select.correct-answer-select').each(function () {
            var answer = $(this).val();
            correctAnswerArray.push(answer);
        });

        var availableCorrectAnswerArray = possibleAnswerArray.filter((item) => !correctAnswerArray.includes(item));

        //clear all existing correct answers
        $('select.correct-answer-select').each(function () {
            $(this).empty();
        });
        var correctAnswerSelect = document.getElementsByClassName('correct-answer-select');
        for (var i = 0; i < correctAnswerSelect.length; i++) {
            //add currently selected option to each element
            var opt = document.createElement('option');
            opt.value = correctAnswerArray[i];
            opt.innerHTML = correctAnswerArray[i];
            correctAnswerSelect.item(i).appendChild(opt);

            //add remaining available options to each element
            for (var j = 0; j < availableCorrectAnswerArray.length; j++) {
                var opt = document.createElement('option');
                opt.value = availableCorrectAnswerArray[j];
                opt.innerHTML = availableCorrectAnswerArray[j];
                correctAnswerSelect.item(i).appendChild(opt);
            }
        }
    }
}

function editCorrectAnswers(action, subquestionType) {
    //users wants to edit correct answers
    if (action == "enable") {
        if (subquestionType != 4 && subquestionType != 9) {
            document.getElementById("possible-answer-edit").disabled = true;
        }
        
        document.getElementById("correct-answer-edit").disabled = true;
        document.getElementById("correct-answer-save").disabled = false;
        document.getElementById("subquestion-add").disabled = true;

        if (subquestionType == 1) {
            $(".correct-answer-move").prop('disabled', false);
        }
        else if (subquestionType == 2) {
            document.getElementById("correct-answer-add").disabled = false;
            $(".correct-answer-delete").prop('disabled', false);
            $('.correct-answer-select').prop('disabled', false);
        }
        else if (subquestionType == 3) {
            $('.correct-answer-select').prop('disabled', false);
        }
        else if (subquestionType == 4) {
            $('.correct-answer-input').prop('disabled', false);
            $('.correct-answer-radio').prop('disabled', false);
            $('.correct-answer-delete').prop('disabled', false);
            document.getElementById("correct-answer-add").disabled = false;
        }
        else if (subquestionType == 6 || subquestionType == 7) {
            $('.correct-answer-select').prop('disabled', false);
        }
        else if (subquestionType == 9) {
            $('.correct-answer-input').prop('disabled', false);
            document.getElementById("subquestion-text-edit").disabled = true;
        } 
        else if (subquestionType == 10) {
            document.getElementById("slider-question").disabled = false;
        }
    }
    //user is done editing correct answers
    else if (action == "disable") {
        var addAnswers = true;
        var correctAnswerList = [];
        if (subquestionType == 2 || subquestionType == 3 || subquestionType == 6 || subquestionType == 7) {
            $('.correct-answer-select').each(function () {
                var answer = $(this).val();
                correctAnswerList.push(answer);
                if (answer == getOptionPlaceholderText()) {
                    alert("Chyba: nevyplněná správná odpověď.");
                    addAnswers = false;
                    return false;
                }
            });
        }
        else if (subquestionType == 4) {
            var seen = {};
            $('.correct-answer-input').each(function () {
                var answer = $(this).val();
                correctAnswerList.push(answer);
                if (answer.length == 0) {
                    alert("Chyba: nevyplněná otázka.");
                    addAnswers = false;
                    return false;
                }
                if (seen[answer]) {
                    alert("Chyba: duplikátní otázka (" + answer + ").");
                    addAnswers = false;
                    return false;
                }
                else {
                    seen[answer] = true;
                }  
            });
        }

        if (addAnswers) {
            if (subquestionType != 4 && subquestionType != 9) {
                document.getElementById("possible-answer-edit").disabled = false;
            }

            document.getElementById("correct-answer-edit").disabled = false;
            document.getElementById("correct-answer-save").disabled = true;
            document.getElementById("subquestion-add").disabled = false;

            if (subquestionType == 1) {
                $(".correct-answer-move").prop('disabled', true);
            }
            else if (subquestionType == 2) {
                document.getElementById("correct-answer-add").disabled = true;
                $(".correct-answer-delete").prop('disabled', true);
                $('.correct-answer-select').prop('disabled', true);
                var subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
                document.getElementById("subquestion-add").disabled = false;
            }
            else if (subquestionType == 3) {
                $('.correct-answer-select').prop('disabled', true);
                var subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
                document.getElementById("subquestion-add").disabled = false;
            }
            else if (subquestionType == 4) {
                $('.correct-answer-input').prop('disabled', true);
                $('.correct-answer-radio').prop('disabled', true);
                $('.correct-answer-delete').prop('disabled', true);
                document.getElementById("correct-answer-add").disabled = true;
            }
            else if (subquestionType == 6 || subquestionType == 7) {
                $('.correct-answer-select').prop('disabled', true);
                var subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
                document.getElementById("subquestion-add").disabled = false;
                if (subquestionType == 7) {
                    document.getElementById("gap-text").value = document.getElementsByClassName("correct-answer-select")[0].value;
                }
            }
            else if (subquestionType == 9) {
                $('.correct-answer-input').prop('disabled', true);
                var subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
                document.getElementById("subquestion-add").disabled = false;
                document.getElementById("subquestion-text-edit").disabled = false;

                var answerNumber = 0;
                $('.gap-text').each(function () {
                    $(this).val(correctAnswerList[answerNumber]);
                    answerNumber++;
                });
            }
            else if (subquestionType == 10) {
                document.getElementById("slider-question").disabled = true;
            }
        }
    }
}

//enables the user to move correct answers (upwards or downwards)
function moveCorrectAnswer(direction, clicked_id) {
    var table = document.getElementById('correct-answers-table');
    var rowIndex = 0;
    for (var i = 0, row; row = table.rows[i]; i++) {
        if (clicked_id == row.id) {
            rowIndex = i;
            break;
        }
    }
    var rowCount = table.rows.length;
    if (direction == 'up') {
        if (rowIndex > 1) {
            var temp = table.rows[rowIndex - 1].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex - 1].cells[0].getElementsByTagName("input")[0].value = table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value = temp;
        }
    }
    else if (direction == 'down') {
        if (rowIndex + 1 < rowCount) {
            var temp = table.rows[rowIndex + 1].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex + 1].cells[0].getElementsByTagName("input")[0].value = table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value = temp;
        }
    }
}

function deleteCorrectAnswer(clicked_id, subquestionType) {
    var table = document.getElementById('correct-answers-table');
    var rowCount = table.rows.length;
    if (rowCount <= '2') {
        if (subquestionType == 4) {
            alert('Chyba: musí existovat alespoň 1 otázka.');
        }
        else {
            alert('Chyba: musí existovat alespoň 1 správná odpověď.');
        }
    }
    else {
        var row = document.getElementById(clicked_id);
        row.parentNode.removeChild(row);
        if (subquestionType != 4) {
            updateCorrectAnswersSelect("correctAnswerChosen", subquestionType);
        }
    }
}

//just before form submission, certain fields have to be modified so that they get properly binded to the SubquestionTemplate
function onAddSubquestionFormSubmission(subquestionType) {
    if (subquestionType == 2 || subquestionType == 6) {
        //correct answer selects are enabled so that they get binded to subquestionTemplate
        $('.correct-answer-select').prop('disabled', false);
    }
    else if (subquestionType == 3) {
        $('.correct-answer-select').prop('disabled', false);
        var correctAnswerArray = [];
        var correctAnswersTable = document.getElementById('correct-answers-table');
        var rowCount = correctAnswersTable.rows.length;
        for (var i = 1; i < rowCount; i++) {
            var firstAnswer = correctAnswersTable.rows[i].cells[0].getElementsByTagName("select")[0].value;
            var secondAnswer = correctAnswersTable.rows[i].cells[0].getElementsByTagName("select")[1].value;
            correctAnswerArray.push(firstAnswer + " -> " + secondAnswer);
        }

        for (var i = 1; i < rowCount; i++) {
            correctAnswersTable.rows[i].cells[0].getElementsByTagName("select")[0].value = correctAnswerArray[i];
            correctAnswersTable.rows[i].cells[0].getElementsByTagName("select")[1].value = correctAnswerArray[i];
        }//todo: edit
    }
    else if (subquestionType == 4) {
        //for this type of subquestion, correct answers must be preprocessed before form submission
        $('.correct-answer-input').prop('disabled', false);
        $('.correct-answer-radio').prop('disabled', false);
        var correctAnswerArray = [];
        $('input[type="text"].correct-answer-input').each(function () {
            var answer = $(this).val();
            correctAnswerArray.push(answer);
        });
        var correctAnswersTable = document.getElementById('correct-answers-table');
        for (var i = 0; i < correctAnswerArray.length; i++) {
            if (correctAnswersTable.rows[i + 1].cells[1].getElementsByTagName("input")[0].checked) {
                correctAnswerArray[i] = correctAnswerArray[i] + " -> Ano";
            }
            else {
                correctAnswerArray[i] = correctAnswerArray[i] + " -> Ne";
            }
        }
        var answerNumber = 0;
        $('input[type="text"].correct-answer-input').each(function () {
            $(this).val(correctAnswerArray[answerNumber]);
            answerNumber++;
        });
    }
    else if (subquestionType == 7 || subquestionType == 8) {
        if (subquestionType == 7) {
            $('.correct-answer-select').prop('disabled', false);
        }
        //for this type of subquestion, subquestion text must be preprocessed before form submission
        var subquestionText = document.getElementById("SubquestionText").value;
        var subquestionTextContinuation = document.getElementById("SubquestionTextContinuation").value;
        subquestionText += " (DOPLŇTE) " + subquestionTextContinuation;
        document.getElementById("SubquestionText").value = subquestionText;
    }
    else if (subquestionType == 9) {
        $('.subquestion-text').prop('disabled', false);
        $('.correct-answer-input').prop('disabled', false);
        var subquestionTexts = document.getElementsByClassName("subquestion-text");
        var subquestionText = "";
        for (var i = 0; i < subquestionTexts.length; i++) {
            subquestionText += subquestionTexts[i].value;
            if (i != subquestionTexts.length - 1) {
                subquestionText += "(DOPLŇTE[" + (i + 1) + "])";
            }
        }
        document.getElementById("SubquestionTextHidden").value = subquestionText;
    }
    else if (subquestionType == 10) {
        var min = document.getElementById("slider-min").value;
        var max = document.getElementById("slider-max").value;
        document.getElementById("possible-answer-list-hidden").value = min + " - " + max;
        document.getElementById("correct-answer-list-hidden").value = document.getElementById("slider-question").value;
    }
}

//after the user changes subquestion points, correct and wrong choice points are updated automatically
function updateChoicePoints(subquestionPoints, subquestionType) {
    const formatter = new Intl.NumberFormat('en-GB', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
    subquestionPoints = subquestionPoints.value;
    var possibleChoiceArrayLength = 0;
    if (subquestionType != 4 && subquestionType != 5 && subquestionType != 8 && subquestionType != 9 && subquestionType != 10) {
        var possibleAnswersTable = document.getElementById('possible-answers-table');
        possibleChoiceArrayLength = possibleAnswersTable.rows.length - 1;
    }
    if (subquestionType != 5 && subquestionType != 8 && subquestionType != 10) {
        var correctAnswersTable = document.getElementById('correct-answers-table');
        var correctChoiceArrayLength = correctAnswersTable.rows.length - 1;
    }

    //check if points can be updated or not
    if (subquestionPoints != null && subquestionPoints != "" &&
        (possibleChoiceArrayLength >= 1 || (subquestionType == 4 || subquestionType == 5 || subquestionType == 8 || subquestionType == 9 || subquestionType == 10)) &&
        (correctChoiceArrayLength >= 1 || subquestionType == 5 || subquestionType == 8 || subquestionType == 10)) {
        var correctChoicePoints = 0;
        switch (subquestionType) {
            case 1:
            case 5:
            case 6:
            case 7:
            case 8:
            case 10:
                correctChoicePoints = subquestionPoints;
                break;
            case 2:
                correctChoicePoints = formatter.format(subquestionPoints) / formatter.format(correctChoiceArrayLength);
                break;
            case 3:
            case 4:
            case 9:
                correctChoicePoints = formatter.format(subquestionPoints) / (formatter.format(correctChoiceArrayLength) / 2) / 2;
                break;
        }

        correctChoicePoints = formatter.format(correctChoicePoints);

        document.getElementById("correct-choice-points").value = correctChoicePoints;
        document.getElementById("wrongChoicePoints_automatic").value = correctChoicePoints * (-1);
        if (subquestionType != 5) {
            document.getElementById("wrongChoicePoints_manual").min = correctChoicePoints * (-1);
        }
    }
}

function setSubquestionTypeDetails(subquestionType) {
    var subquestionTypeDetailsArray = [
        "Neznámý nebo nepodporovaný typ otázky!",
        "Úkolem je seřadit pojmy v daném pořadí (např. od nejnižšího po nejvyšší, od nejmenšího po největší).",
        "Úkolem je z daných možných odpovědí vybrat jednu nebo více správných odpovědí.",
        "Úkolem je spojit nějakým způsobem související pojmy do dvojic.",
        "Úkolem je odpovědět Ano / Ne u několika různých pojmů v rámci jedné podotázky.",
        "Úkolem je volně odpovědět na otázku do textového pole.",
        "Úkolem je z daných možných odpovědí vybrat jednu správnou odpověď.",
        "Úkolem je z daných možných odpovědí doplnit do věty jednu správnou odpověď.",
        "Úkolem je doplnit do věty správnou odpověď, přičemž student nevybírá z možností.",
        "Úkolem je dané možné odpovědi doplnit na správná místa ve větách.",
        "Úkolem je zvolit z posuvníku správnou odpověď (v číselné formě)."
    ];
    var index = subquestionType.selectedIndex;
    document.getElementById("subquestion-type-details").innerHTML = subquestionTypeDetailsArray[index + 1];
}

function removeImage() {
    document.getElementById("imagePath").value = "";
}

function fillGapText(correctAnswerInput) {
    document.getElementById("gap-text").value = correctAnswerInput.value;
}

//adds another gap (another question) to the subquestion text of subquestion of type 9
function addGap() {
    var br = document.createElement("br");
    var additionalQuestions = document.getElementById("additional-questions");
    additionalQuestions.appendChild(br);

    var gapTexts = document.getElementsByClassName("gap-text");
    var gapText = gapTexts[gapTexts.length - 1];
    var clonedGapText = gapText.cloneNode(true);
    clonedGapText.value = "[" + parseInt(gapTexts.length + 1) + "] - (DOPLŇTE)";
    additionalQuestions.appendChild(clonedGapText);

    var subquestionTexts = document.getElementsByClassName("subquestion-text");
    var subquestionText = subquestionTexts[subquestionTexts.length - 1];
    var clonedSubquestionText = subquestionText.cloneNode(true);
    clonedSubquestionText.value = "";
    clonedSubquestionText.placeholder = parseInt(subquestionTexts.length + 1) + ". část věty";
    additionalQuestions.appendChild(clonedSubquestionText);
    additionalQuestions.appendChild(br);
}

function removeGap() {
    var additionalQuestions = document.getElementById("additional-questions");
    var gapTexts = additionalQuestions.getElementsByClassName("gap-text");
    var gapText = gapTexts[gapTexts.length - 1];
    if (gapTexts.length > 0) {//only remove gap in case more than 1 gap exists
        additionalQuestions.removeChild(gapText);

        var subquestionTexts = additionalQuestions.getElementsByClassName("subquestion-text");
        var subquestionText = subquestionTexts[subquestionTexts.length - 1];
        additionalQuestions.removeChild(subquestionText);

        var brs = additionalQuestions.getElementsByTagName("br");
        additionalQuestions.removeChild(brs[brs.length - 1]);
    }
}

function editSubquestionText(action) {
    //users wants to edit subquestion text
    if (action == "enable") {
        $('.subquestion-text').prop('disabled', false);
        document.getElementById("gap-add").disabled = false;
        document.getElementById("gap-remove").disabled = false;
        document.getElementById("subquestion-text-edit").disabled = true;
        document.getElementById("subquestion-text-save").disabled = false;
        document.getElementById("correct-answer-edit").disabled = true;
    }
    //user is done editing subquestion text
    else if (action == "disable") {
        var addAnswers = true;
        $('.subquestion-text').each(function () {
            var answer = $(this).val();
            if (answer.length == 0) {
                alert("Chyba: nevyplněná správná odpověď.");
                addAnswers = false;
                return false;
            } 
        });

        if(addAnswers){
            $('.subquestion-text').prop('disabled', true);
            updateCorrectAnswersInputFreeAnswer();
            document.getElementById("correct-answer-edit").disabled = false;
            document.getElementById("gap-add").disabled = true;
            document.getElementById("gap-remove").disabled = true;
            document.getElementById("subquestion-text-edit").disabled = false;
            document.getElementById("subquestion-text-save").disabled = true;
            document.getElementById("correct-answer-edit").disabled = false;
        }
    }
}

//function that is called after the AddSubquestionTemplate page is loaded - disables or enables certain fields, adds possible answer rows..
function addSubquestionTemplatePagePostProcessing(subquestionType) {
    if (subquestionType != 0)
    {
        document.getElementById("subquestionType").selectedIndex = parseInt(subquestionType - 1);
    }

    if (subquestionType == 1) {
        addPossibleAnswer();
    }
    else if (subquestionType == 2) {
        addPossibleAnswer();
    }
    else if (subquestionType == 3) {
        addPossibleAnswer();
        addPossibleAnswer();
        addPossibleAnswer();
    }
    else if (subquestionType == 5) {
        document.getElementById("wrongChoicePoints_manual").disabled = true;
        document.getElementById("subquestion-add").disabled = false;
    }
    else if (subquestionType == 6) {
        addPossibleAnswer();
    }
    else if (subquestionType == 7) {
        addPossibleAnswer();
        document.getElementById("SubquestionText").placeholder = "První část věty";
    }
    else if (subquestionType == 8) {
        document.getElementById("SubquestionText").placeholder = "První část věty";
        document.getElementById("subquestion-add").disabled = false;
    }
    else if (subquestionType == 9) {
        document.getElementById("SubquestionText").outerHTML = "";
    }
}

function pointsRecommendationPostProcessing(subquestionType, possibleAnswerListString, correctAnswerListString) {
    var possibleAnswerTable = document.getElementById('possible-answers-table');
    var possibleAnswerTableLength = document.getElementById('possible-answers-table').rows.length;
    var possibleAnswerList = [];
    var possibleAnswerListStringSplit = possibleAnswerListString.split(";");
    for (var i = 0; i < possibleAnswerListStringSplit.length; i++) {
        possibleAnswerList.push(possibleAnswerListStringSplit[i]);
    }
    var possibleAnswerListLength = possibleAnswerList.length;

    if (possibleAnswerListLength > possibleAnswerTableLength) {
        var difference = possibleAnswerListLength - possibleAnswerTableLength;
        for (var i = 0; i < difference; i++) {
            addPossibleAnswer();
        }
    }

    if (subquestionType == 1 || subquestionType == 2 || subquestionType == 3) {
        for (var i = 1; i < possibleAnswerListLength; i++) {
            possibleAnswerTable.rows[i].cells[0].getElementsByTagName("input")[0].value = possibleAnswerList[i - 1];
        }
    }

    var correctAnswerTable = document.getElementById('correct-answers-table');
    var correctAnswerTableLength = document.getElementById('correct-answers-table').rows.length;
    var correctAnswerList = [];
    var correctAnswerListStringSplit = correctAnswerListString.split(";");
    for (var i = 0; i < correctAnswerListStringSplit.length; i++) {
        correctAnswerList.push(correctAnswerListStringSplit[i]);
    }
    var correctAnswerListLength = correctAnswerList.length;

    if (subquestionType == 1 || subquestionType == 2 || subquestionType == 3) {
        if (correctAnswerListLength > correctAnswerTableLength) {
            var difference = correctAnswerListLength - correctAnswerTableLength;
            for (var i = 0; i < difference; i++) {
                addCorrectAnswer(subquestionType, true);
            }
        }
    }

    if (subquestionType == 1) {
        for (var i = 1; i < correctAnswerListLength; i++) {
            correctAnswerTable.rows[i].cells[0].getElementsByTagName("input")[0].value = correctAnswerList[i - 1];
        }
    }
    else if (subquestionType == 2 || subquestionType == 3) {
        var correctAnswerSelects = document.getElementsByClassName('correct-answer-select');
        for (var i = 0; i < correctAnswerListLength - 1; i++) {
            var opt = document.createElement('option');
            opt.value = correctAnswerList[i];
            opt.innerHTML = correctAnswerList[i];
            correctAnswerSelects.item(i).appendChild(opt);
        }
    }

    $('.correct-answer-select').prop('disabled', true);
}

//General

function hideConfirmActionForm() {
    document.getElementById("confirm-action").style.display = "none";
}