//ManageUserRegistrationList.cshtml

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

const adminForm = document.getElementById('admin-form');
adminForm.addEventListener('submit', adminFormSubmit);

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
        document.getElementById("possible-answer-add").disabled = false;
        document.getElementById("possible-answer-edit").disabled = true;
        document.getElementById("correct-answer-edit").disabled = true;
        document.getElementById("subquestion-add").disabled = true;
        $(".possible-answer-delete").prop('disabled', false);
        document.getElementById("possible-answer-save").disabled = false;
        $(".possible-answer-input").prop('readonly', false);
        $(".possible-answer-move").prop('disabled', false);
    }
    //user is done editing possible answers
    else if (action == "disable") {
        var addAnswers = true;
        var seen = {};
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

        if (addAnswers) {
            if (subquestionType == 1) {
                updateCorrectAnswersInput();
                document.getElementById("subquestion-add").disabled = false;
            }
            else if (subquestionType == 2) {
                updateCorrectAnswersSelect("possibleAnswersModified");
            }
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
    }
}

function deletePossibleAnswer(clicked_id) {
    var table = document.getElementById('possible-answers-table');
    var rowCount = table.rows.length;
    if (rowCount <= '3') {
        alert('Chyba: musí existovat alespoň 2 možné odpovědi.');
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

function addCorrectAnswer() {
    //check if new correct answer can be added
    var possibleAnswersTable = document.getElementById('possible-answers-table');
    var possibleAnswersTableRowCount = possibleAnswersTable.rows.length;
    var correctAnswersTable = document.getElementById('correct-answers-table');
    var correctAnswersTableRowCount = correctAnswersTable.rows.length;

    if (correctAnswersTableRowCount >= possibleAnswersTableRowCount) {
        alert('Chyba: může existovat maximálně ' + (possibleAnswersTableRowCount - 1) + " možných odpovědí.");
    }
    else {
        var rowCount = correctAnswersTable.rows.length;
        var lastRowInnerHTML = correctAnswersTable.rows[rowCount - 1].innerHTML;
        var lastRowIdArray = correctAnswersTable.rows[rowCount - 1].id.split("-");
        var lastRowId = parseInt(lastRowIdArray[2]);
        lastRowId += 1;
        var row = correctAnswersTable.insertRow(rowCount);
        row.innerHTML = lastRowInnerHTML;
        row.id = "correct-answer-" + lastRowId;

        //replace currently selected option with placeholder option
        var correctAnswerSelect = document.getElementsByClassName('correct-answer-select');
        correctAnswerSelect[correctAnswerSelect.length - 1].options[0].innerHTML = getOptionPlaceholderText();
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

    for (var i = 0; i < possibleAnswerArray.length - 2; i++) {
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

//automatic update of correct answers when dropdown menus are used after possible answers are modified
//subquestion types - 2
function updateCorrectAnswersSelect(performedAction) {
    //user modified possible answers - all correct answers are deleted and replaced by new possible answers
    if (performedAction == "possibleAnswersModified") {
        var possibleAnswerArray = [];
        possibleAnswerArray.push(getOptionPlaceholderText());
        $('input[type="text"].possible-answer-input').each(function () {
            var answer = $(this).val();
            possibleAnswerArray.push(answer);
        });

        //clear all existing correct answers
        $('select.correct-answer-select').each(function () {
            $(this).empty();
        });

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
            if (answer != getOptionPlaceholderText()) {
                correctAnswerArray.push(answer);
            }
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

            /*if (correctAnswerArray[i] != "-ZVOLTE MOŽNOST-") {
                //add placeholder option to each element
                var opt = document.createElement('option');
                opt.value = "-ZVOLTE MOŽNOST-";
                opt.innerHTML = "-ZVOLTE MOŽNOST-";
                correctAnswerSelect.item(i).appendChild(opt);
            }*/

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
        document.getElementById("possible-answer-edit").disabled = true;
        document.getElementById("correct-answer-edit").disabled = true;
        document.getElementById("correct-answer-save").disabled = false;
        $(".correct-answer-move").prop('disabled', false);
        document.getElementById("subquestion-add").disabled = true;
        if (subquestionType == 2) {
            document.getElementById("correct-answer-add").disabled = false;
            $(".correct-answer-delete").prop('disabled', false);
            $('.correct-answer-select').prop('disabled', false);
        }
    }
    //user is done editing correct answers
    else if (action == "disable") {
        var addAnswers = true;
        var correctAnswerList = [];
        $('.correct-answer-select').each(function () {
            var answer = $(this).val();
            correctAnswerList.push(answer);
            if (answer == getOptionPlaceholderText()) {
                alert("Chyba: nevyplněná správná odpověď.");
                addAnswers = false;
                return false;
            }
        });

        if (addAnswers) {
            document.getElementById("possible-answer-edit").disabled = false;
            document.getElementById("correct-answer-edit").disabled = false;
            document.getElementById("correct-answer-save").disabled = true;
            $(".correct-answer-move").prop('disabled', true);
            document.getElementById("subquestion-add").disabled = false;
            if (subquestionType == 2) {
                document.getElementById("correct-answer-add").disabled = true;
                $(".correct-answer-delete").prop('disabled', true);
                $('.correct-answer-select').prop('disabled', true);
                var subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
                document.getElementById("subquestion-add").disabled = false;

                //because the correct answer selects are disabled during form submission, a hidden field must be used instead to bind correct answers
                document.getElementById("hidden-correct-answer-list").value = correctAnswerList;
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

function deleteCorrectAnswer(clicked_id) {
    var table = document.getElementById('correct-answers-table');
    var rowCount = table.rows.length;
    if (rowCount <= '2') {
        alert('Chyba: musí existovat alespoň 1 správná odpověď.');
    }
    else {
        var row = document.getElementById(clicked_id);
        row.parentNode.removeChild(row);
        updateCorrectAnswersSelect("correctAnswerChosen");
    }
}


//after the user changes subquestion points, correct and wrong choice points are updated automatically
function updateChoicePoints(subquestionPoints, subquestionType) {
    const formatter = new Intl.NumberFormat('en-GB', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
    subquestionPoints = subquestionPoints.value;
    var possibleAnswersTable = document.getElementById('possible-answers-table');
    var possibleChoiceArrayLength = possibleAnswersTable.rows.length - 1;
    var correctAnswersTable = document.getElementById('correct-answers-table');
    var correctChoiceArrayLength = correctAnswersTable.rows.length - 1;

    //check if points can be updated or not
    if (subquestionPoints != null && subquestionPoints != "" && possibleChoiceArrayLength >= 1 && correctChoiceArrayLength >= 1) {
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
        document.getElementById("wrongChoicePoints_manual").min = correctChoicePoints * (-1);
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
    console.log(index);
    document.getElementById("subquestion-type-details").innerHTML = subquestionTypeDetailsArray[index + 1];
}

function removeImage() {
    document.getElementById("imagePath").value = "";
}

//General

function hideConfirmActionForm() {
    document.getElementById("confirm-action").style.display = "none";
}