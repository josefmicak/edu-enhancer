//Index.cshtml

/**
 * Loads option groups and their members on home screen in case testing mode is turned on
 * @param {string} login User login
 * @param {string} email User email
 * @param {string} role User role
 * @param {string} mainAdminCount Amount of main admins
 * @param {string} adminCount Amount of admins
 * @param {string} teacherCount Amount of teachers
 */
function optGroupUpdate(login, email, role, mainAdminCount, adminCount, teacherCount) {
    let opt = document.createElement('OPTION');
    opt.textContent = "Email: " + email + ", login: " + login;
    opt.value = login;
    if (role == "Teacher") {
        document.getElementById("teacher-optgroup").appendChild(opt);
        document.getElementById("teacher-optgroup").setAttribute('label', 'Učitelé (' + teacherCount + ")");
    }
    else if (role == "Admin") {
        document.getElementById("admin-optgroup").appendChild(opt);
        document.getElementById("admin-optgroup").setAttribute('label', 'Správci (' + adminCount + ")");
    }
    else if (role == "MainAdmin") {
        document.getElementById("main-admin-optgroup").appendChild(opt);
        document.getElementById("main-admin-optgroup").setAttribute('label', 'Hlavní administrátoři (' + mainAdminCount + ")");
    }
}

//ManageUserRegistrationList.cshtml

/**
 * Updates registration tables according to user's settings - can hide accepted/rejected registrations
 * @param {boolean} accepted Whether user wants to see accepted registrations
 * @param {boolean} rejected Whether user wants to see rejected registrations
 * @param {boolean} text Whether user has just changed his settings
 */
function registrationsTableUpdate(accepted, rejected, text) {
    if (accepted == false) {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[7].textContent == "Schválena") {
                tr.style.display = 'none';
            }
            if (tr.children[8] != null && tr.children[8].textContent == "Schválena") {
                tr.style.display = 'none';
            }
        });
    }
    else {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[7].textContent == "Schválena") {
                tr.style.display = '';
            }
            if (tr.children[8] != null && tr.children[8].textContent == "Schválena") {
                tr.style.display = '';
            }
        });
    }

    if (rejected == false) {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[7].textContent == "Zamítnuta") {
                tr.style.display = 'none';
            }
            if (tr.children[8] != null && tr.children[8].textContent == "Zamítnuta") {
                tr.style.display = 'none';
            }
        });
    }
    else {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[7].textContent == "Zamítnuta") {
                tr.style.display = '';
            }
            if (tr.children[8] != null && tr.children[8].textContent == "Zamítnuta") {
                tr.style.display = '';
            }
        });
    }

    if (text == true) {
        document.getElementById('visibility-text').innerHTML = 'Změny úspěšně uloženy.';
    }
}

//ManageUserList.cshtml / ManageUserListForAdmin.cshtml

/**
 * Shown after user chooses to edit student's details
 * @param {string} oldLogin Student's login
 * @param {string} email Student's email
 * @param {string} firstName Student's first name
 * @param {string} lastName Student's last name
 */
function showEditStudentLabel(oldLogin, email, firstName, lastName) {
    document.getElementById("student-action").value = 'editStudent';
    document.getElementById("added-student").style.display = 'none';
    document.getElementById("edited-student").style.display = 'block';
    document.getElementById("studentOldLogin").value = oldLogin;
    document.getElementById("studentFirstName").value = firstName;
    document.getElementById('studentFirstName').readOnly = false;
    document.getElementById("studentLastName").value = lastName;
    document.getElementById('studentLastName').readOnly = false;
    document.getElementById("studentLogin").value = oldLogin;
    document.getElementById('studentLogin').readOnly = true;
    document.getElementById("studentEmail").value = email;
}

/**
 * Shown after user chooses to not edit student's details (after previous choosing to edit them)
 */
function hideEditStudentLabel() {
    document.getElementById("student-action").value = 'addStudent';
    document.getElementById("added-student").style.display = 'block';
    document.getElementById("edited-student").style.display = 'none';
    document.getElementById("studentFirstName").value = "";
    document.getElementById("studentLastName").value = "";
    document.getElementById("studentLogin").value = "";
    document.getElementById('studentLogin').readOnly = false;
    document.getElementById("studentEmail").value = "";
}

/**
 * Shown after user chooses to edit teacher's details
 * @param {string} oldLogin Teacher's login
 * @param {string} email Teacher's email
 * @param {string} firstName Teacher's first name
 * @param {string} lastName Teacher's last name
 * @param {boolean} makeVisible Whether option to change teacher's role is shown (it's not possible for admins to change a teacher's role)
 */
function showEditTeacherLabel(oldLogin, email, firstName, lastName, makeVisible) {
    document.getElementById("teacher-action").value = 'editTeacher';
    document.getElementById("added-teacher").style.display = 'none';
    document.getElementById("edited-teacher").style.display = 'block';
    document.getElementById("teacherOldLogin").value = oldLogin;
    document.getElementById("teacherFirstName").value = firstName;
    document.getElementById("teacherLastName").value = lastName;
    document.getElementById("teacherLogin").value = oldLogin;
    document.getElementById('teacherLogin').readOnly = true;
    document.getElementById("teacherEmail").value = email;

    if (makeVisible) {
        document.getElementById("teacher-edit-role").style.display = 'block';
    }
}

/**
 * Shown after user chooses to not edit teacher's details (after previously choosing to edit them)
 */
function hideEditTeacherLabel() {
    document.getElementById("teacher-action").value = 'addTeacher';
    document.getElementById("added-teacher").style.display = 'block';
    document.getElementById("edited-teacher").style.display = 'none';
    document.getElementById("teacher-edit-role").style.display = 'none';
    document.getElementById("teacherFirstName").value = "";
    document.getElementById("teacherLastName").value = "";
    document.getElementById("teacherLogin").value = "";
    document.getElementById('teacherLogin').readOnly = false;
    document.getElementById("teacherEmail").value = "";
}

/**
 *  Shown after user chooses to edit admin's details
 * @param {string} oldLogin Admin's login
 * @param {string} email Admin's email
 * @param {string} firstName Admin's first name
 * @param {string} lastName Admin's last name
 * @param {string} role Admin's role (admin/mail admin)
 */
function showEditAdminLabel(oldLogin, email, firstName, lastName, role) {
    document.getElementById("admin-action").value = 'editAdmin';
    document.getElementById("added-admin").style.display = 'none';
    document.getElementById("edited-admin").style.display = 'block';
    document.getElementById("admin-edit-role").style.display = 'block';
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

/**
 * Shown after user chooses to not edit admin's details (after previously choosing to edit them)
 */
function hideEditAdminLabel() {
    document.getElementById("admin-action").value = 'addAdmin';
    document.getElementById("added-admin").style.display = 'block';
    document.getElementById("edited-admin").style.display = 'none';
    document.getElementById("admin-edit-role").style.display = 'none';
    document.getElementById("adminFirstName").value = "";
    document.getElementById("adminLastName").value = "";
    document.getElementById("adminLogin").value = "";
    document.getElementById('adminLogin').readOnly = false;
    document.getElementById("adminEmail").value = "";
}

/**
 * Disables or enables dropdown with available roles
 * @param {object} checkbox Checkbox where main admin selects whether he wants to change admin's role or not
 */
function changeAdminRole(checkbox) {
    const checked = checkbox.checked;
    if (checked) {
        document.getElementById("change-admin-role").disabled = false;
    }
    else {
        document.getElementById("change-admin-role").disabled = true;
    }
}

/**
 * Triggered after main admin chooses to select other user as main admin (he's asked for confirmation)
 * @param {object} event Event that gets triggered after form submission
 */
function adminFormSubmit(event) {
    let isMainAdmin = document.getElementById("isMainAdmin").value;
    let role = document.getElementById("change-admin-role").value;

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

document.addEventListener('DOMContentLoaded', function (event) {
    //check if admin-form exists to prevent invalid JS element call
    let elementExists = document.getElementById("admin-form");
    if (elementExists) {
        const adminForm = document.getElementById('admin-form');
        adminForm.addEventListener('submit', adminFormSubmit);
    }
});

//QuestionTemplate.cshtml

/**
 * Disables and enables input fields concerned with wrong choice points after user clicks a radiobutton connected with wrong choice points
 * @param {object} el Clicked wrong choice points radiobutton
 */
function setWrongChoicePointsInputs(el) {
    if (el.value == "wrongChoicePoints_automatic_radio") {
        document.getElementById("wrongChoicePoints_automatic").disabled = false;
        document.getElementById("wrongChoicePoints_manual").disabled = true;
    }
    else {
        document.getElementById("wrongChoicePoints_automatic").disabled = true;
        document.getElementById("wrongChoicePoints_manual").disabled = false;
    }
}

/**
 * Called after the QuestionTemplate page is loaded - edits certain fields, changes selects..
 * @param {number} subquestionNumber Subquestion index
 * @param {number} subquestionsCount Amount of subquestions within a question
 * @param {string} subquestionElementName Subquestion result ID
 */
function questionTemplatePagePostProcessing(subquestionNumber, subquestionsCount, subquestionElementName) {
    document.getElementById(subquestionElementName).selectedIndex = subquestionNumber;
    if (subquestionNumber == 0 || subquestionsCount == 1) {
        document.getElementById("previousSubquestion").disabled = true;
    }
    if ((subquestionNumber == subquestionsCount - 1) || subquestionsCount <= 1) {
        document.getElementById("nextSubquestion").disabled = true;
    }

    modifyInputNumbers();
}

/**
 * Shown to user after the user tries to perform a potentially dangerous action - user has to confirm action
 * @param {string} action Action to be performed
 * @param {string} identifier Identifier of the modified entity
 * @param {string} email User email
 * @param {string} login User login
 * @param {string} firstName User first name
 * @param {string} lastName User last name
 * @param {string} role User role 
 */
function showConfirmActionForm(action, identifier, email, login, firstName, lastName, role) {
    document.getElementById("confirm-action").style.display = "block";
    document.getElementById("action").value = action;
    if (action == "deleteTemplate") {
        document.getElementById("testTemplateId").value = identifier;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit tuto testovou šablonu?";
    }
    else if (action == "deleteAllTemplates") {
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit všechny testovací šablony?";
    }
    else if (action == "deleteQuestionTemplate") {
        document.getElementById("questionTemplateId").value = identifier;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit toto zadání otázky?";
    }
    else if (action == "deleteSubquestionTemplate") {
        document.getElementById("subquestionTemplateIdToDelete").value = identifier;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit toto zadání podotázky?";
    }
    else if (action == "deleteResult") {
        document.getElementById("testResultId").value = identifier;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit toto řešení testu?";
    }
    else if (action == "deleteAllResults") {
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit všechna řešení testů?";
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
    else if (action == "deleteSubject") {
        document.getElementById("subjectId").value = identifier;
        document.getElementById("confirm-action-label").innerHTML = "Opravdu si přejete odstranit tento předmět?" +
            " (provedením této akce budou smazány všechny testy patřící pod tento předmět).";
    }
}

//AddSubquestionTemplate.cshtml

/**
 * Placeholder text for options - has to be uniform within the application
 */
function getOptionPlaceholderText() {
    return '-ZVOLTE MOŽNOST-';
}

/**
 * Adds possible answer (can be done only with certain subquestion types)
 */
function addPossibleAnswer() {
    let table = document.getElementById('possible-answers-table');
    let rowCount = table.rows.length;
    let lastRowInnerHTML = table.rows[rowCount - 1].innerHTML;
    let lastRowIdArray = table.rows[rowCount - 1].id.split("-");
    let lastRowId = parseInt(lastRowIdArray[2]);
    lastRowId += 1;
    let row = table.insertRow(rowCount);
    row.innerHTML = lastRowInnerHTML;
    row.id = "possible-answer-" + lastRowId;
}

/**
 * Called after user either chooses to edit possible answers or after he is finished editing possible answers
 * @param {string} action Action to be performed (enable - start editing, disable - stop editing)
 * @param {number} subquestionType Type of the subqestion
 */
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
        let addAnswers = true;
        let seen = {};
        let min = 0;
        let max = 0;

        $('input[type="text"].possible-answer-input').each(function () {
            let answer = $(this).val();
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
            if (parseInt(min) >= parseInt(max)) {
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
                let subquestionPoints = document.getElementById("subquestion-points");
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
                let subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
            }
            else if (subquestionType == 10) {
                $(".slider-input").prop('disabled', true);
                let sliderQuestion = document.getElementById("slider-question");
                sliderQuestion.min = min;
                sliderQuestion.max = max;
                sliderQuestion.value = Math.round((parseInt(min) + parseInt(max)) / 2);
                sliderQuestion.parentNode.nextElementSibling.value = Math.round((parseInt(min) + parseInt(max)) / 2);
                document.getElementById("possible-answer-save").disabled = true;
                document.getElementById("possible-answer-edit").disabled = false;
                document.getElementById("correct-answer-edit").disabled = false;
                document.getElementById("subquestion-add").disabled = false;

                let subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
            }
        }
    }
}

/**
 * Deletes possible answer (can be done only with certain subquestion types)
 * @param {number} clicked_id ID of the deleted possible answer
 * @param {number} subquestionType Type of the subqestion
 */
function deletePossibleAnswer(clicked_id, subquestionType) {
    let minPossibleAnswers = [0, 2, 2, 4, 0, 0, 2, 2, 0, 2, 0];
    let table = document.getElementById('possible-answers-table');
    let rowCount = table.rows.length;
    if (rowCount <= minPossibleAnswers[subquestionType] + 1) {
        alert('Chyba: musí existovat alespoň ' + minPossibleAnswers[subquestionType] + ' možné odpovědi.');
    }
    else {
        let row = document.getElementById(clicked_id);
        row.parentNode.removeChild(row);
    }
}

/**
 * Moves possible answer (upwards or downwards)
 * @param {string} direction Direction (upwards or downwards)
 * @param {number} clicked_id ID of the selected answer
 */
function moveAnswer(direction, clicked_id) {
    let table = document.getElementById('possible-answers-table');
    let rowIndex = 0;
    for (let i = 0, row; row = table.rows[i]; i++) {
        if (clicked_id == row.id) {
            rowIndex = i;
            break;
        }
    }
    let rowCount = table.rows.length;
    if (direction == 'up') {
        if (rowIndex > 1) {
            let temp = table.rows[rowIndex - 1].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex - 1].cells[0].getElementsByTagName("input")[0].value = table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value = temp;
        }
    }
    else if (direction == 'down') {
        if (rowIndex + 1 < rowCount) {
            let temp = table.rows[rowIndex + 1].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex + 1].cells[0].getElementsByTagName("input")[0].value = table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value = temp;
        }
    }
}

/**
 * Adds correct answer (can be done only with certain subquestion types)
 * @param {number} subquestionType Type of the subquestion
 * @param {boolean} isProgramatical Whether user himself initiated this action or whether it's been initiated automatically (e.g. after page load)
 */
function addCorrectAnswer(subquestionType, isProgramatical) {
    let addAnswer = true;
    let correctAnswersTable = document.getElementById('correct-answers-table');

    if (subquestionType == 2) {
        //check if new correct answer can be added
        let possibleAnswersTable = document.getElementById('possible-answers-table');
        let possibleAnswersTableRowCount = possibleAnswersTable.rows.length;
        let correctAnswersTableRowCount = correctAnswersTable.rows.length;

        if (correctAnswersTableRowCount >= possibleAnswersTableRowCount) {
            addAnswer = false;
            alert('Chyba: může existovat maximálně ' + (possibleAnswersTableRowCount - 1) + " možných odpovědí.");
        }
    }

    if (addAnswer) {
        let rowCount = correctAnswersTable.rows.length;
        let lastRowInnerHTML = correctAnswersTable.rows[rowCount - 1].innerHTML;
        let lastRowIdArray = correctAnswersTable.rows[rowCount - 1].id.split("-");
        let lastRowId = parseInt(lastRowIdArray[2]);
        lastRowId += 1;

        if (subquestionType == 1 || subquestionType == 9) {//programmer use only
            let row = correctAnswersTable.insertRow(rowCount);
            row.innerHTML = lastRowInnerHTML;
            row.id = "correct-answer-" + lastRowId;
        }
        else if (subquestionType == 2 || subquestionType == 3) {
            let row = correctAnswersTable.insertRow(rowCount);
            row.innerHTML = lastRowInnerHTML;
            row.id = "correct-answer-" + lastRowId;

            //replace currently selected option with placeholder option
            if (!isProgramatical) {
                let correctAnswerSelects = document.getElementsByClassName('correct-answer-select');
                correctAnswerSelects[correctAnswerSelects.length - 1].options[0].innerHTML = getOptionPlaceholderText();
            }
        }
        else if (subquestionType == 4) {
            let lastRowRadioNameArray = correctAnswersTable.rows[rowCount - 1].cells[1].getElementsByTagName("input")[0].name.split("-");
            let lastRowRadioNumber = parseInt(lastRowRadioNameArray[3]);

            let yesChecked = true;//incicates whether the "yes" option is checked on the last row
            if (correctAnswersTable.rows[rowCount - 1].cells[2].getElementsByTagName("input")[0].checked) {
                yesChecked = false;
            }

            let row = correctAnswersTable.insertRow(rowCount);
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

/**
 * Updates correct answers after user changes possible answers (only subquestion type 1)
 */
function updateCorrectAnswersInput() {
    let possibleAnswerArray = [];
    possibleAnswerArray.push(getOptionPlaceholderText());
    $('input[type="text"].possible-answer-input').each(function () {
        let answer = $(this).val();
        possibleAnswerArray.push(answer);
    });

    //clear correct answers table
    let table = document.getElementById('correct-answers-table');
    let rowCount = table.rows.length;
    while (--rowCount - 1) {
        table.deleteRow(rowCount);
    }
    table.rows[1].cells[0].getElementsByTagName("input")[0].value = "";

    for (let i = 0; i < possibleAnswerArray.length - 2; i++) {
        rowCount = table.rows.length;
        let lastRowInnerHTML = table.rows[rowCount - 1].innerHTML;
        let lastRowIdArray = table.rows[rowCount - 1].id.split("-");
        let lastRowId = parseInt(lastRowIdArray[2]);
        lastRowId += 1;
        let row = table.insertRow(rowCount);
        row.innerHTML = lastRowInnerHTML;
        row.id = "correct-answer-" + lastRowId;
    }

    for (let i = 1; i < possibleAnswerArray.length; i++) {
        table.rows[i].cells[0].getElementsByTagName("input")[0].value = possibleAnswerArray[i];
    }
}

/**
 * Updates correct answers after user changes possible answers (only subquestion type 1)
 * Adds an appropriate number of correct answers
 */
function updateCorrectAnswersInputFreeAnswer() {
    let additionalQuestions = document.getElementById("additional-questions");
    let table = document.getElementById('correct-answers-table');
    let rowCount = table.rows.length;
    while (--rowCount - 1) {
        table.deleteRow(rowCount);
    }
    table.rows[1].cells[0].getElementsByTagName("input")[0].value = "";
    table.rows[1].cells[0].getElementsByTagName("input")[0].placeholder = "[1] - Správná odpověď";

    let additionalGapTexts = additionalQuestions.getElementsByClassName("gap-text");
    for (let i = 0; i < additionalGapTexts.length; i++) {
        rowCount = table.rows.length;
        let lastRowInnerHTML = table.rows[rowCount - 1].innerHTML;
        let lastRowIdArray = table.rows[rowCount - 1].id.split("-");
        let lastRowId = parseInt(lastRowIdArray[2]);
        lastRowId += 1;
        let row = table.insertRow(rowCount);
        row.innerHTML = lastRowInnerHTML;
        row.id = "correct-answer-" + lastRowId;
        row.cells[0].getElementsByTagName("input")[0].placeholder = "[" + parseInt(i + 2) + "] - Správná odpověď";
    }

    let gapTexts = document.getElementsByClassName("gap-text");
    for (let i = 0; i < gapTexts.length; i++) {
        gapTexts[i].value = "[" + (i + 1) + "] - (DOPLŇTE)";
    }
}

/**
 * Updates correct answers after user changes possible answers (subquestion types 2, 3, 6)
 * Dropdown menus are updated
 * @param {string} performedAction possibleAnswersModified / correctAnswerChosen
 * @param {number} subquestionType Type of the subquestion
 */
function updateCorrectAnswersSelect(performedAction, subquestionType) {
    //user modified possible answers - all correct answers are deleted and replaced by new possible answers
    if (performedAction == "possibleAnswersModified") {
        let possibleAnswerArray = [];
        possibleAnswerArray.push(getOptionPlaceholderText());
        $('input[type="text"].possible-answer-input').each(function () {
            let answer = $(this).val();
            possibleAnswerArray.push(answer);
        });

        //clear correct answers table
        let table = document.getElementById('correct-answers-table');
        let rowCount = table.rows.length;
        while (--rowCount - 1) {
            table.deleteRow(rowCount);
        }
        $(".correct-answer-select").empty();

        //for some subquestion types, new rows must be added to the correct answer table
        if (subquestionType == 3) {
            //let table = document.getElementById('correct-answers-table');
            for (let i = 0; i < Math.floor((possibleAnswerArray.length - 3) / 2); i++) {//-3 because of 1 already existing row and 1 possible answer containing placeholder text
                rowCount = table.rows.length;
                let lastRowInnerHTML = table.rows[rowCount - 1].innerHTML;
                let lastRowIdArray = table.rows[rowCount - 1].id.split("-");
                let lastRowId = parseInt(lastRowIdArray[2]);
                lastRowId += 1;
                let row = table.insertRow(rowCount);
                row.innerHTML = lastRowInnerHTML;
                row.id = "correct-answer-" + lastRowId;
            }
        }

        let correctAnswerSelect = document.getElementsByClassName('correct-answer-select');
        for (let i = 0; i < correctAnswerSelect.length; i++) {
            for (let j = 0; j < possibleAnswerArray.length; j++) {
                let opt = document.createElement('option');
                opt.value = possibleAnswerArray[j];
                opt.innerHTML = possibleAnswerArray[j];
                correctAnswerSelect.item(i).appendChild(opt);
            }
        }

        let correctAnswerArray = [];
        $('select.correct-answer-select').each(function () {
            let answer = $(this).val();
            correctAnswerArray.push(answer);
        });
    }
    //user selected or deleted a correct answer
    //this selected or deleted answer is removed from all other dropdowns, while previously selected answer is added to all other dropdowns
    else if (performedAction == "correctAnswerChosen") {
        let possibleAnswerArray = [];
        $('input[type="text"].possible-answer-input').each(function () {
            let answer = $(this).val();
            possibleAnswerArray.push(answer);
        });

        let correctAnswerArray = [];
        $('select.correct-answer-select').each(function () {
            let answer = $(this).val();
            correctAnswerArray.push(answer);
        });

        let availableCorrectAnswerArray = possibleAnswerArray.filter((item) => !correctAnswerArray.includes(item));

        //clear all existing correct answers
        $('select.correct-answer-select').each(function () {
            $(this).empty();
        });
        let correctAnswerSelect = document.getElementsByClassName('correct-answer-select');
        for (let i = 0; i < correctAnswerSelect.length; i++) {
            //add currently selected option to each element
            let opt = document.createElement('option');
            opt.value = correctAnswerArray[i];
            opt.innerHTML = correctAnswerArray[i];
            correctAnswerSelect.item(i).appendChild(opt);

            //add remaining available options to each element
            for (let j = 0; j < availableCorrectAnswerArray.length; j++) {
                let optRemaining = document.createElement('option');
                optRemaining.value = availableCorrectAnswerArray[j];
                optRemaining.innerHTML = availableCorrectAnswerArray[j];
                correctAnswerSelect.item(i).appendChild(optRemaining);
            }
        }
    }
}

/**
 * Called after user either chooses to edit correct answers or after he is finished editing correct answers
 * @param {string} action Action to be performed (enable - start editing, disable - stop editing)
 * @param {number} subquestionType Type of the subqestion
 */
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
        let addAnswers = true;
        let correctAnswerList = [];
        if (subquestionType == 2 || subquestionType == 3 || subquestionType == 6 || subquestionType == 7) {
            $('.correct-answer-select').each(function () {
                let answer = $(this).val();
                correctAnswerList.push(answer);
                if (answer == getOptionPlaceholderText()) {
                    alert("Chyba: nevyplněná správná odpověď.");
                    addAnswers = false;
                    return false;
                }
            });
        }
        else if (subquestionType == 4 || subquestionType == 9) {
            let seen = {};
            $('.correct-answer-input').each(function () {
                let answer = $(this).val();
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
                let subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
                document.getElementById("subquestion-add").disabled = false;
            }
            else if (subquestionType == 3) {
                $('.correct-answer-select').prop('disabled', true);
                let subquestionPoints = document.getElementById("subquestion-points");
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
                let subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
                document.getElementById("subquestion-add").disabled = false;
                if (subquestionType == 7) {
                    document.getElementById("gap-text").value = document.getElementsByClassName("correct-answer-select")[0].value;
                }
            }
            else if (subquestionType == 9) {
                $('.correct-answer-input').prop('disabled', true);
                let subquestionPoints = document.getElementById("subquestion-points");
                updateChoicePoints(subquestionPoints, subquestionType);
                document.getElementById("subquestion-add").disabled = false;
                document.getElementById("subquestion-text-edit").disabled = false;

                let answerNumber = 0;
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

/**
 * Moves correct answer (upwards or downwards)
 * @param {string} direction Direction (upwards or downwards)
 * @param {number} clicked_id ID of the selected answer
 */
function moveCorrectAnswer(direction, clicked_id) {
    let table = document.getElementById('correct-answers-table');
    let rowIndex = 0;
    for (let i = 0, row; row = table.rows[i]; i++) {
        if (clicked_id == row.id) {
            rowIndex = i;
            break;
        }
    }
    let rowCount = table.rows.length;
    if (direction == 'up') {
        if (rowIndex > 1) {
            let temp = table.rows[rowIndex - 1].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex - 1].cells[0].getElementsByTagName("input")[0].value = table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value = temp;
        }
    }
    else if (direction == 'down') {
        if (rowIndex + 1 < rowCount) {
            let temp = table.rows[rowIndex + 1].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex + 1].cells[0].getElementsByTagName("input")[0].value = table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value;
            table.rows[rowIndex].cells[0].getElementsByTagName("input")[0].value = temp;
        }
    }
}

/**
 * Deletes correct answer (can be done only with certain subquestion types)
 * @param {number} clicked_id ID of the deleted possible answer
 * @param {number} subquestionType Type of the subqestion
 */
function deleteCorrectAnswer(clicked_id, subquestionType) {
    let table = document.getElementById('correct-answers-table');
    let rowCount = table.rows.length;
    if (rowCount <= '2') {
        if (subquestionType == 4) {
            alert('Chyba: musí existovat alespoň 1 otázka.');
        }
        else {
            alert('Chyba: musí existovat alespoň 1 správná odpověď.');
        }
    }
    else {
        let row = document.getElementById(clicked_id);
        row.parentNode.removeChild(row);
        if (subquestionType != 4) {
            updateCorrectAnswersSelect("correctAnswerChosen", subquestionType);
        }
    }
}

/**
 * After user chooses to add subquestion, certain fields are modified just before form submission in order to get properly binded to the SubquestionTemplate
 * @param {number} subquestionType Type of the subquestion
 */
function onAddSubquestionFormSubmission(subquestionType) {
    let suggestedPointsLabel = document.getElementById("suggested-points-label");
    suggestedPointsLabel.innerHTML = "Doporučený počet bodů za otázku: probíhá výpočet..";
    let suggestedPointsButton = document.getElementById("suggested-points-button");
    suggestedPointsButton.style.display = "none";

    if (subquestionType == 2 || subquestionType == 6) {
        //correct answer selects are enabled so that they get binded to subquestionTemplate
        $('.correct-answer-select').prop('disabled', false);
    }
    else if (subquestionType == 3) {
        $('.correct-answer-select').prop('disabled', false);
    }
    else if (subquestionType == 4) {
        //for this type of subquestion, correct answers must be preprocessed before form submission
        $('.correct-answer-input').prop('disabled', false);
        $('.correct-answer-radio').prop('disabled', false);
        let correctAnswerArray = [];

        let correctAnswersTable = document.getElementById('correct-answers-table');
        for (let i = 0; i < correctAnswersTable.rows.length - 1; i++) {
            if (correctAnswersTable.rows[i + 1].cells[1].getElementsByTagName("input")[0].checked) {
                correctAnswerArray.push("1");
            }
            else {
                correctAnswerArray.push("0");
            }
        }
        let answerNumber = 0;
        $('.correct-answer-hidden').each(function () {
            $(this).val(correctAnswerArray[answerNumber]);
            answerNumber++;
        });
    }
    else if (subquestionType == 7 || subquestionType == 8) {
        if (subquestionType == 7) {
            $('.correct-answer-select').prop('disabled', false);
        }
    }
    else if (subquestionType == 9) {
        $('.subquestion-text').prop('disabled', false);
        $('.correct-answer-input').prop('disabled', false);
    }
    else if (subquestionType == 10) {
        let sliderValues = [];
        let min = document.getElementById("slider-min").value;
        let max = document.getElementById("slider-max").value;
        let sliderQuestion = document.getElementById("slider-question").value;
        sliderValues.push(min);
        sliderValues.push(max);
        sliderValues.push(sliderQuestion);
        document.getElementById("sliderValues").value = sliderValues;
    }
}

/**
 * After the user changes subquestion points, correct and wrong choice points are updated automatically
 * @param {object} subquestionPoints New value of subquestion points
 * @param {number} subquestionType Type of the subquestion
 */
function updateChoicePoints(subquestionPoints, subquestionType) {
    subquestionPoints = subquestionPoints.value;
    let possibleChoiceArrayLength = 0;
    if (subquestionType != 4 && subquestionType != 5 && subquestionType != 8 && subquestionType != 9 && subquestionType != 10) {
        let possibleAnswersTable = document.getElementById('possible-answers-table');
        possibleChoiceArrayLength = possibleAnswersTable.rows.length - 1;
    }
    let correctChoiceArrayLength = 0;
    if (subquestionType != 5 && subquestionType != 8 && subquestionType != 10) {
        let correctAnswersTable = document.getElementById('correct-answers-table');
        correctChoiceArrayLength = correctAnswersTable.rows.length - 1;
    }

    //check if points can be updated or not
    if (subquestionPoints != null && subquestionPoints != "" &&
        (possibleChoiceArrayLength >= 1 || (subquestionType == 4 || subquestionType == 5 || subquestionType == 8 || subquestionType == 9 || subquestionType == 10)) &&
        (correctChoiceArrayLength >= 1 || subquestionType == 5 || subquestionType == 8 || subquestionType == 10)) {
        let correctChoicePoints = 0;
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
                correctChoicePoints = (Number.parseFloat(subquestionPoints.replace(",", ".")) / correctChoiceArrayLength).toFixed(2);
                break;
            case 3:
            case 4:
            case 9:
                correctChoicePoints = (Number.parseFloat(subquestionPoints.replace(",", ".")) / (correctChoiceArrayLength / 2) / 2).toFixed(2);
                break;
        }

        document.getElementById("correct-choice-points").value = correctChoicePoints.toString().replace(".", ",");
        document.getElementById("wrongChoicePoints_automatic").value = ("-" + correctChoicePoints).toString().replace(".", ",");
    }
}

/**
 * Changes value of the label that describes the purpose of the selected subquestion type
 * @param {number} subquestionType Type of the subquestion
 */
function setSubquestionTypeDetails(subquestionType) {
    let subquestionTypeDetailsArray = [
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
    if (document.getElementById("subquestion-type-details") != null) {
        document.getElementById("subquestion-type-details").innerHTML = subquestionTypeDetailsArray[subquestionType + 1];
    }
}

/**
 * Removes image when the user is adding or editing subquestion
 */
function removeImage() {
    document.getElementById("imagePath").value = "";
    document.getElementById("ImageSource").value = "";
    fileLabel.innerHTML = "Obrázek nebyl vybrán.";
}

/**
 * Fills gap between first and second part of the question sentence with correct answer after user changes the answer
 * @param {object} correctAnswerInput Input field of the correct answer
 */
function fillGapText(correctAnswerInput) {
    document.getElementById("gap-text").value = correctAnswerInput.value;
}

/**
 * Adds another gap (another question) to the subquestion text of subquestion of type 9
 */
function addGap() {
    let br = document.createElement("br");
    let additionalQuestions = document.getElementById("additional-questions");
    additionalQuestions.appendChild(br);

    let gapTexts = document.getElementsByClassName("gap-text");
    let gapText = gapTexts[gapTexts.length - 1];
    let clonedGapText = gapText.cloneNode(true);
    clonedGapText.value = "[" + parseInt(gapTexts.length + 1) + "] - (DOPLŇTE)";
    additionalQuestions.appendChild(clonedGapText);

    let subquestionTexts = document.getElementsByClassName("subquestion-text");
    let subquestionText = subquestionTexts[subquestionTexts.length - 1];
    let clonedSubquestionText = subquestionText.cloneNode(true);
    clonedSubquestionText.value = "";
    clonedSubquestionText.placeholder = parseInt(subquestionTexts.length + 1) + ". část věty";
    additionalQuestions.appendChild(clonedSubquestionText);
    additionalQuestions.appendChild(br);
}

/**
 * Removes gap from the subquestion text of subquestion of type 9
 */
function removeGap() {
    let additionalQuestions = document.getElementById("additional-questions");
    let gapTexts = additionalQuestions.getElementsByClassName("gap-text");
    let gapText = gapTexts[gapTexts.length - 1];
    if (gapTexts.length > 1) {//only remove gap in case more than 2 gaps exist
        additionalQuestions.removeChild(gapText);

        let subquestionTexts = additionalQuestions.getElementsByClassName("subquestion-text");
        let subquestionText = subquestionTexts[subquestionTexts.length - 1];
        additionalQuestions.removeChild(subquestionText);

        let brs = additionalQuestions.getElementsByTagName("br");
        additionalQuestions.removeChild(brs[brs.length - 1]);
    }
}

/**
 * Called after user either chooses to edit subquestion text or after he is finished editing subquestion text
 * @param {string} action Action to be performed (enable - start editing, disable - stop editing)
 */
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
        let addAnswers = true;
        $('.subquestion-text').each(function () {
            let answer = $(this).val();
            if (answer.length == 0) {
                alert("Chyba: nevyplněná otázka.");
                addAnswers = false;
                return false;
            }
        });

        if (addAnswers) {
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

/**
 * Called after the AddSubquestionTemplate/EditSubquestionTemplate page is loaded - disables or enables certain fields, adds possible answer rows..
 * @param {number} subquestionType Type of the subquestion that is being added
 * @param {boolean} changeIndex Whether index should be changed - true if AddSubquestionTemplate, false if EditSubquestionTemplate
 */
function addSubquestionTemplatePagePostProcessing(subquestionType, changeIndex) {
    if (subquestionType != 0 && changeIndex) {
        if (document.getElementById("subquestionType") != null) {
            document.getElementById("subquestionType").selectedIndex = parseInt(subquestionType - 1);
        }
        setSubquestionTypeDetails(subquestionType - 1);
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
        document.getElementById("SubquestionText").outerHTML = "";
    }
    else if (subquestionType == 8) {
        document.getElementById("SubquestionText").outerHTML = "";
        document.getElementById("subquestion-add").disabled = false;
    }
    else if (subquestionType == 9) {
        document.getElementById("SubquestionText").outerHTML = "";
        addGap();
        updateCorrectAnswersInputFreeAnswer();
    }

    modifyInputNumbers();
}

/**
 * Called after user adds or removes image on the AddSubquestionTemplate/EditSubquestionTemplate page
 * @param {string} imageSource Source of the image
 */
function changeImagePath(imageSource) {
    let imagePath = document.getElementById('imagePath');
    if (imageSource != null) {
        fileLabel.innerHTML = imageSource;
        document.getElementById("ImageSource").value = imageSource;
    }
    else if (imagePath.value == "") {
        fileLabel.innerHTML = "Obrázek nebyl vybrán";
        document.getElementById("ImageSource").value = "";
    }
    else {
        let theSplit = imagePath.value.split('\\');
        fileLabel.innerHTML = theSplit[theSplit.length - 1];
        document.getElementById("ImageSource").value = theSplit[theSplit.length - 1];
    }
}

/**
 * Called after the user requests points recommendation in the AddSubquestionTemplate page or if he visits the EditSubquestionTemplate page
 * @param {number} subquestionType Type of the subquestion
 * @param {string} possibleAnswerListString List of possible answers, separated by ";"
 * @param {string} correctAnswerListString List of correct answers, separated by ";"
 * @param {string} subquestionText Text of the subquestion
 * @param {number} defaultWrongChoicePoints Default wrong choice points of the subquestion
 * @param {number} wrongChoicePoints Wrong choice points of the subquestion
 */
function pointsRecommendationPostProcessing(subquestionType, possibleAnswerListString, correctAnswerListString, subquestionText,
    defaultWrongChoicePoints, wrongChoicePoints) {
    let possibleAnswerTable = null;
    let correctAnswerTable = null;
    let possibleAnswerListLength = 0;
    let correctAnswerTableLength = 0;
    let possibleAnswerList = [];

    if (subquestionType == 1 || subquestionType == 2 || subquestionType == 3 || subquestionType == 6 || subquestionType == 7) {
        possibleAnswerTable = document.getElementById('possible-answers-table');
        let possibleAnswerTableLength = document.getElementById('possible-answers-table').rows.length;
        let possibleAnswerListStringSplit = possibleAnswerListString.split(";");
        for (let i = 0; i < possibleAnswerListStringSplit.length; i++) {
            possibleAnswerList.push(possibleAnswerListStringSplit[i]);
        }
        possibleAnswerListLength = possibleAnswerList.length;

        if (possibleAnswerListLength > possibleAnswerTableLength) {
            let difference = possibleAnswerListLength - possibleAnswerTableLength;
            for (let i = 0; i < difference; i++) {
                addPossibleAnswer();
            }
        }
    }
    else if (subquestionType == 4 || subquestionType == 10) {
        let possibleAnswerListStringSplit = possibleAnswerListString.split(";");
        for (let i = 0; i < possibleAnswerListStringSplit.length; i++) {
            possibleAnswerList.push(possibleAnswerListStringSplit[i]);
        }
        possibleAnswerListLength = possibleAnswerList.length;
    }

    if (subquestionType == 1 || subquestionType == 2 || subquestionType == 3 || subquestionType == 6 || subquestionType == 7) {
        for (let i = 1; i < possibleAnswerListLength; i++) {
            possibleAnswerTable.rows[i].cells[0].getElementsByTagName("input")[0].value = possibleAnswerList[i - 1];
        }
    }

    if (subquestionType != 5 && subquestionType != 8 && subquestionType != 10) {
        correctAnswerTable = document.getElementById('correct-answers-table');
        correctAnswerTableLength = document.getElementById('correct-answers-table').rows.length;
    }
    let correctAnswerList = [];
    let correctAnswerListStringSplit = correctAnswerListString.split(";");
    if (subquestionType == 3) {
        for (let i = 0; i < correctAnswerListStringSplit.length; i++) {
            let splitAgain = correctAnswerListStringSplit[i].split("|");
            correctAnswerList.push(splitAgain[0]);
            correctAnswerList.push(splitAgain[1]);
        }
    }
    else {
        for (let i = 0; i < correctAnswerListStringSplit.length; i++) {
            correctAnswerList.push(correctAnswerListStringSplit[i]);
        }
    }
    let correctAnswerListLength = correctAnswerList.length;

    if (subquestionType == 1 || subquestionType == 2 || subquestionType == 4 || subquestionType == 9) {
        if (correctAnswerListLength > correctAnswerTableLength) {
            let difference = correctAnswerListLength - correctAnswerTableLength;
            for (let i = 0; i < difference; i++) {
                addCorrectAnswer(subquestionType, true);
            }
        }
    }
    else if (subquestionType == 3) {
        if (correctAnswerListLength > correctAnswerTableLength) {
            let difference = (correctAnswerListLength / 2) - correctAnswerTableLength;
            for (let i = 0; i < difference; i++) {
                addCorrectAnswer(subquestionType, true);
            }
        }
    }

    if (subquestionType == 1 || subquestionType == 9) {
        for (let i = 1; i < correctAnswerListLength; i++) {
            correctAnswerTable.rows[i].cells[0].getElementsByTagName("input")[0].value = correctAnswerList[i - 1];
        }
    }
    else if (subquestionType == 2 || subquestionType == 3 || subquestionType == 6 || subquestionType == 7) {
        let correctAnswerSelects = document.getElementsByClassName('correct-answer-select');
        for (let i = 0; i < correctAnswerListLength - 1; i++) {
            let opt = document.createElement('option');
            opt.value = correctAnswerList[i];
            opt.innerHTML = correctAnswerList[i];
            correctAnswerSelects.item(i).appendChild(opt);
        }
    }
    else if (subquestionType == 4) {
        for (let i = 1; i < correctAnswerListLength; i++) {
            correctAnswerTable.rows[i].cells[0].getElementsByTagName("input")[0].value = possibleAnswerList[i - 1];
            if (correctAnswerList[i - 1] == "1") {
                correctAnswerTable.rows[i].cells[1].getElementsByTagName("input")[0].checked = true;
            }
            else if (correctAnswerList[i - 1] == "0") {
                correctAnswerTable.rows[i].cells[2].getElementsByTagName("input")[0].checked = true;
            }
        }
    }

    if (subquestionType != 4 && subquestionType != 5 && subquestionType != 8 && subquestionType != 9 && subquestionType != 10) {
        $('.correct-answer-select').prop('disabled', true);
    }

    //in case the subquestion text includes gaps, we must divide it and assign it to appropriate gaps
    if (subquestionType == 7 || subquestionType == 8) {
        let subquestionTextSplit = subquestionText.split("|");
        document.getElementsByName("subquestionTextArray[]")[0].value = subquestionTextSplit[0];
        document.getElementById("gap-text").value = correctAnswerList[0];
        document.getElementsByName("subquestionTextArray[]")[1].value = subquestionTextSplit[1];

        if (subquestionType == 8) {
            document.getElementById("correct-answer-input").value = correctAnswerList[0];
        }
    }
    else if (subquestionType == 9) {
        let subquestionTextSplit = subquestionText.split("|");
        for (let i = 0; i < subquestionTextSplit.length; i++) {
            if (i > 0 && i < subquestionTextSplit.length - 2) {//due to gap that exists by default and empty string
                addGap();
            }
            document.getElementsByClassName("subquestion-text")[i].value = subquestionTextSplit[i];
            if (i != subquestionTextSplit.length - 1) {//there is one less gap than subquestion text input
                document.getElementsByClassName("gap-text")[i].value = correctAnswerList[i];
            }
        }
    }

    if (subquestionType == 10) {
        document.getElementById("slider-min").value = possibleAnswerList[0];
        document.getElementById("slider-max").value = possibleAnswerList[1];
        let sliderQuestion = document.getElementById("slider-question");
        sliderQuestion.min = possibleAnswerList[0];
        sliderQuestion.max = possibleAnswerList[1];
        sliderQuestion.value = correctAnswerList[0];
        sliderQuestion.parentNode.nextElementSibling.value = correctAnswerList[0];
    }

    //set manual wrong choice points radio to checked in case it was checked before
    if (defaultWrongChoicePoints != wrongChoicePoints) {
        document.getElementById("wrongChoicePoints_manual_radio").checked = true;
        document.getElementById("wrongChoicePoints_manual").disabled = false;
    }

    modifyInputNumbers();
}

//AddSubject.cshtml

/**
 * User adds selected students to the subject
 */
function addStudentsToSubject() {
    let unenrolledStudentsTable = document.getElementById("unenrolled-students-table");
    let enrolledStudentsTable = document.getElementById("enrolled-students-table");

    for (let i = 1; i < unenrolledStudentsTable.rows.length; i++) {
        if (unenrolledStudentsTable.rows[i].cells[2].getElementsByTagName("input")[0].checked == true) {
            let unenrolledRowInnerHTML = unenrolledStudentsTable.rows[i].innerHTML;
            let enrolledRow = enrolledStudentsTable.insertRow(enrolledStudentsTable.rows.length);
            enrolledRow.innerHTML = unenrolledRowInnerHTML;
            enrolledRow.cells[1].getElementsByTagName("input")[0].name = "enrolledStudentLogin[]";

            unenrolledStudentsTable.deleteRow(i);
            i--;
        }
    }
}

/**
 * User removes selected students from the subject
 */
function removeStudentsFromSubject() {
    let unenrolledStudentsTable = document.getElementById("unenrolled-students-table");
    let enrolledStudentsTable = document.getElementById("enrolled-students-table");

    for (let i = 1; i < enrolledStudentsTable.rows.length; i++) {
        if (enrolledStudentsTable.rows[i].cells[2].getElementsByTagName("input")[0].checked == true) {
            let enrolledRowInnerHTML = enrolledStudentsTable.rows[i].innerHTML;
            let unenrolledRow = unenrolledStudentsTable.insertRow(unenrolledStudentsTable.rows.length);
            unenrolledRow.innerHTML = enrolledRowInnerHTML;
            unenrolledRow.cells[1].getElementsByTagName("input")[0].name = "unenrolledStudentLogin[]";

            enrolledStudentsTable.deleteRow(i);
            i--;
        }
    }
}

//SolveQuestion.cshtml

/**
 * Called after the SolveQuestion page is loaded
 * @param {number} subquestionsCount Amount of subquestions in the test
 * @param {number} subquestionResultIdIndex Index of current subquestion within the text
 * @param {number} subquestionType Type of the current subquestion
 * @param {string} possibleAnswerListString List of possible answers, separated by ";"
 * @param {string} correctAnswerListString List of correct answers, separated by ";"
 * @param {string} subquestionText Text of the current subquestion
 * @param {string} studentAnswerListString List of student's answers, separated by ";"
 * @param {string} answerCompletenessString List of completeness of all student's answers (whether the student has answered the subquestions or not), separated by ";"
 */
function solveQuestionPagePostProcessing(subquestionsCount, subquestionResultIdIndex, subquestionType, possibleAnswerListString,
    correctAnswerListString, subquestionText, studentAnswerListString, answerCompletenessString) {
    let studentAnswerTable = null;
    let possibleAnswerList = [];
    let possibleAnswerListLength = 0;
    let unshuffledPossibleAnswers = [];
    let parentNodeId = null;

    //disable buttons in case first or last subquestion is shown
    if (subquestionResultIdIndex == 0) {
        document.getElementById("previousSubquestion").disabled = true;
    }
    if (subquestionResultIdIndex == subquestionsCount - 1) {
        document.getElementById("nextSubquestion").disabled = true;
    }

    //add rows to student's answers table
    if (subquestionType == 1 || subquestionType == 2 || subquestionType == 3 || subquestionType == 4 || subquestionType == 6) {
        studentAnswerTable = document.getElementById('student-answers-table');
        let studentAnswerTableLength = document.getElementById('student-answers-table').rows.length;
        let possibleAnswerListStringSplit = possibleAnswerListString.split(";");
        for (let i = 0; i < possibleAnswerListStringSplit.length - 1; i++) {
            possibleAnswerList.push(possibleAnswerListStringSplit[i]);
        }
        for (let i = 0; i < possibleAnswerList.length; i++) {
            unshuffledPossibleAnswers.push(possibleAnswerList[i]);
        }
        possibleAnswerList = shuffleArray(possibleAnswerList);
        possibleAnswerListLength = possibleAnswerList.length + 1;

        if (subquestionType == 1) {
            if (possibleAnswerListLength > studentAnswerTableLength) {
                let difference = possibleAnswerListLength - studentAnswerTableLength;
                for (let i = 0; i < difference; i++) {
                    addStudentAnswer(subquestionType);
                }

                //add rows to possible answers table
                difference = possibleAnswerListLength - studentAnswerTableLength;
                for (let i = 0; i < difference; i++) {
                    addPossibleAnswer();
                }
            }
        }
        else if (subquestionType == 2 || subquestionType == 4 || subquestionType == 6) {
            if (possibleAnswerListLength > studentAnswerTableLength) {
                let difference = possibleAnswerListLength - studentAnswerTableLength;
                for (let i = 0; i < difference; i++) {
                    addStudentAnswer(subquestionType);
                }
            }
        }
        else if (subquestionType == 3) {
            if (possibleAnswerListLength > studentAnswerTableLength) {
                let difference = (possibleAnswerListLength / 2) - studentAnswerTableLength;
                for (let i = 0; i < difference; i++) {
                    addStudentAnswer(subquestionType);
                }

                //add rows to possible answers table
                difference = possibleAnswerListLength - studentAnswerTableLength;
                for (let i = 0; i < difference; i++) {
                    addPossibleAnswer();
                }
            }
        }

    }

    if (subquestionType == 1 || subquestionType == 3) {
        let possibleAnswerTexts = document.getElementsByClassName("possible-answer-text");
        let studentAnswerSelects = document.getElementsByClassName("student-answer-select");
        possibleAnswerList.unshift(getOptionPlaceholderText());

        for (let i = 1; i < possibleAnswerListLength; i++) {
            possibleAnswerTexts.item(i - 1).innerHTML = possibleAnswerList[i];
        }

        for (let i = 0; i < possibleAnswerListLength; i++) {
            for (let j = 0; j < studentAnswerSelects.length; j++) {
                let opt = document.createElement('option');
                opt.value = possibleAnswerList[i];
                opt.innerHTML = possibleAnswerList[i];
                studentAnswerSelects.item(j).appendChild(opt);
            }
        }
    }
    else if (subquestionType == 2 || subquestionType == 6) {
        for (let i = 1; i < possibleAnswerListLength; i++) {
            studentAnswerTable.rows[i].cells[1].innerHTML = possibleAnswerList[i - 1];
        }
    }
    else if (subquestionType == 4) {
        let possibleAnswerTexts = document.getElementsByClassName("possible-answer-text");
        for (let i = 0; i < possibleAnswerTexts.length; i++) {
            possibleAnswerTexts.item(i).innerHTML = possibleAnswerList[i];
        }
    }
    else if (subquestionType == 7 || subquestionType == 8) {
        //add subquestion text
        document.getElementById("SubquestionText").outerHTML = "";
        let subquestionTextSplit = subquestionText.split("|");
        document.getElementById("subquestion-text-1").innerHTML = subquestionTextSplit[0];
        document.getElementById("subquestion-text-2").innerHTML = subquestionTextSplit[1];

        if (subquestionType == 7) {
            //add possible answers to select
            possibleAnswerList.push(getOptionPlaceholderText());
            let possibleAnswerListStringSplit = possibleAnswerListString.split(";");
            for (let i = 0; i < possibleAnswerListStringSplit.length - 1; i++) {
                possibleAnswerList.push(possibleAnswerListStringSplit[i]);
            }

            let studentAnswerSelect = document.getElementById('student-answer-select');
            for (let i = 0; i < possibleAnswerList.length; i++) {
                let opt = document.createElement('option');
                opt.value = possibleAnswerList[i];
                opt.innerHTML = possibleAnswerList[i];
                studentAnswerSelect.appendChild(opt);
            }
            studentAnswerSelect.selectedIndex = 0;
        }
    }
    else if (subquestionType == 9) {
        let possibleAnswerListStringSplit = correctAnswerListString.split(";");
        for (let i = 0; i < possibleAnswerListStringSplit.length - 1; i++) {
            possibleAnswerList.push(possibleAnswerListStringSplit[i]);
        }
        possibleAnswerList = shuffleArray(possibleAnswerList);
        for (let i = 0; i < possibleAnswerList.length - 1; i++) {
            addPossibleAnswer();
        }
        let possibleAnswerTexts = document.getElementsByClassName("possible-answer-text");
        for (let i = 0; i < possibleAnswerTexts.length; i++) {
            possibleAnswerTexts.item(i).innerHTML = possibleAnswerList[i];
        }

        document.getElementById("SubquestionText").outerHTML = "";

        for (let i = 0; i < possibleAnswerList.length - 2; i++) {
            addStudentGap();
        }

        let subquestionTextSplit = subquestionText.split("|");
        let subquestionTexts = document.getElementsByClassName("subquestion-text");
        for (let i = 0; i < subquestionTexts.length; i++) {
            subquestionTexts.item(i).innerHTML = subquestionTextSplit[i];
        }

        possibleAnswerList.unshift(getOptionPlaceholderText());
        let studentAnswerSelects = document.getElementsByClassName("student-answer-select");
        for (let i = 0; i < studentAnswerSelects.length; i++) {
            for (let j = 0; j < possibleAnswerList.length; j++) {
                let opt = document.createElement('option');
                opt.value = possibleAnswerList[j];
                opt.innerHTML = possibleAnswerList[j];
                studentAnswerSelects.item(i).appendChild(opt);
            }
        }
    }
    else if (subquestionType == 10) {
        let possibleAnswerListStringSplit = possibleAnswerListString.split(";");
        let min = possibleAnswerListStringSplit[0];
        let max = possibleAnswerListStringSplit[1];

        let sliderQuestion = document.getElementById("slider-question");
        sliderQuestion.min = min;
        sliderQuestion.max = max;
        sliderQuestion.value = Math.round((parseInt(min) + parseInt(max)) / 2);
    }

    //process student's answer (in case he had already answered this subquestion and has returned to it)
    let studentAnswerList = [];
    if (studentAnswerListString != "") {
        let studentAnswerListStringSplit = studentAnswerListString.split(";");//last item of studentAnswerListStringSplit is empty string
        for (let i = 0; i < studentAnswerListStringSplit.length - 1; i++) {
            if (studentAnswerListStringSplit[i] != getOptionPlaceholderText()) {
                studentAnswerList.push(studentAnswerListStringSplit[i]);
            }
        }
    }

    if (subquestionType == 1) {
        let studentAnswerSelects = document.getElementsByClassName("student-answer-select");
        for (let i = 0; i < studentAnswerList.length; i++) {
            studentAnswerSelects.item(i).value = studentAnswerList[i];
            parentNodeId = "student-answer-" + (i + 1);
            updateStudentsAnswersSelect(parentNodeId, subquestionType);
        }
    }
    else if (subquestionType == 2) {
        let table = document.getElementById('student-answers-table');
        for (let i = 1; i < table.rows.length; i++) {
            let answer = table.rows[i].cells[1].innerHTML;
            if (studentAnswerList.includes(answer)) {
                table.rows[i].cells[0].getElementsByTagName("input")[0].checked = true;
            }
        }
    }
    else if (subquestionType == 3) {
        let studentAnswerSplitByVerticalBar = [];
        let studentAnswerSelects = document.getElementsByClassName("student-answer-select");

        for (let i = 0; i < studentAnswerList.length; i++) {
            let studentAnswerListSplit = studentAnswerList[i].split("|");
            studentAnswerSplitByVerticalBar.push(studentAnswerListSplit[0]);
            studentAnswerSplitByVerticalBar.push(studentAnswerListSplit[1]);
        }
        let studentAnswerId = 1;
        for (let i = 0; i < studentAnswerSplitByVerticalBar.length; i++) {
            studentAnswerSelects.item(i).value = studentAnswerSplitByVerticalBar[i];
            parentNodeId = "student-answer-" + studentAnswerId;
            updateStudentsAnswersSelect(parentNodeId, subquestionType);
            if (i % 2 == 1) {
                studentAnswerId++;
            }
        }
    }
    else if (subquestionType == 4) {
        let table = document.getElementById('student-answers-table');
        for (let i = 0; i < unshuffledPossibleAnswers.length; i++) {
            for (let j = 1; j < table.rows.length; j++) {
                if (unshuffledPossibleAnswers[i] == table.rows[j].cells[0].getElementsByTagName("div")[0].innerHTML) {
                    if (studentAnswerList[i] == "1") {
                        table.rows[j].cells[1].getElementsByTagName("input")[0].checked = true;
                    }
                    else if (studentAnswerList[i] == "0") {
                        table.rows[j].cells[2].getElementsByTagName("input")[0].checked = true;
                    }
                    else if (studentAnswerList[i] != "1" && studentAnswerList[i] != "0") {
                        table.rows[j].cells[1].getElementsByTagName("input")[0].checked = false;
                        table.rows[j].cells[2].getElementsByTagName("input")[0].checked = false;
                    }
                }
            }
        }
    }
    else if (subquestionType == 5) {
        document.getElementById("student-answer").innerText = studentAnswerListString.slice(0, -1);//remove semicolon
    }
    else if (subquestionType == 6) {
        let table = document.getElementById('student-answers-table');
        for (let i = 1; i < table.rows.length; i++) {
            if (studentAnswerListString.slice(0, -1) == table.rows[i].cells[1].innerHTML) {
                table.rows[i].cells[0].getElementsByTagName("input")[0].checked = true;
                break;
            }
        }
    }
    else if (subquestionType == 7) {
        let studentAnswerSelect = document.getElementById('student-answer-select');
        if (studentAnswerListString != "") {
            studentAnswerSelect.value = studentAnswerListString.slice(0, -1);//remove semicolon
        }
    }
    else if (subquestionType == 8) {
        let gapText = document.getElementById('gap-text');
        if (studentAnswerListString != "") {
            gapText.value = studentAnswerListString.slice(0, -1);//remove semicolon
        }
    }
    else if (subquestionType == 9) {
        let studentAnswerSelects = document.getElementsByClassName("student-answer-select");
        for (let i = 0; i < studentAnswerList.length; i++) {
            if (studentAnswerList[i] == "|") {
                studentAnswerSelects.item(i).value = getOptionPlaceholderText();
                updateStudentsAnswersSelect(parentNodeId, subquestionType);
            }
            else {
                studentAnswerSelects.item(i).value = studentAnswerList[i];
                updateStudentsAnswersSelect(parentNodeId, subquestionType);
            }
        }
    }
    else if (subquestionType == 10) {
        if (studentAnswerListString != "") {
            let slider = document.getElementById("slider-question");
            let value = studentAnswerListString.slice(0, -1);//remove semicolon
            slider.value = value
            slider.dispatchEvent(new Event("input"));
            changeSliderOutputs(value);
        }
    }

    addTestNavigationTableElements(answerCompletenessString, subquestionResultIdIndex);
}

/**
 * Adds another field (answer) for the student to fill after the SolveQuestion page is loaded
 * @param {string} subquestionType Type of the subquestion
 */
function addStudentAnswer(subquestionType) {
    let table = document.getElementById('student-answers-table');
    let rowCount = table.rows.length;
    let lastRowInnerHTML = table.rows[rowCount - 1].innerHTML;
    let lastRowId = 0;

    if (subquestionType == 1 || subquestionType == 2 || subquestionType == 3 || subquestionType == 6) {
        let lastRowIdArray = table.rows[rowCount - 1].id.split("-");
        lastRowId = parseInt(lastRowIdArray[2]);
        lastRowId += 1;
        let row = table.insertRow(rowCount);
        row.innerHTML = lastRowInnerHTML;
        row.id = "student-answer-" + lastRowId;
    }
    else if (subquestionType == 4) {
        let lastRowRadioNameArray = table.rows[rowCount - 1].cells[1].getElementsByTagName("input")[0].name.split("-");
        let lastRowRadioNumber = parseInt(lastRowRadioNameArray[3]);

        let row = table.insertRow(rowCount);
        row.innerHTML = lastRowInnerHTML;
        row.id = "student-answer-" + lastRowId;
        row.cells[1].getElementsByTagName("input")[0].name = "student-answer-radio-" + parseInt(lastRowRadioNumber + 1);
        row.cells[2].getElementsByTagName("input")[0].name = "student-answer-radio-" + parseInt(lastRowRadioNumber + 1);

        row.cells[1].getElementsByTagName("input")[0].checked = false;
        row.cells[2].getElementsByTagName("input")[0].checked = false;
    }
}

/**
 * Adds another gap (answer) for the student to fill after the SolveQuestion page is loaded
 * Used only for subquestion of type 9
 */
function addStudentGap() {
    let additionalQuestions = document.getElementById("additional-questions");

    let studentAnswerSelects = document.getElementsByClassName("student-answer-select");
    let studentAnswerSelect = studentAnswerSelects[studentAnswerSelects.length - 1];
    let clonedStudentAnswerSelect = studentAnswerSelect.cloneNode(true);
    additionalQuestions.appendChild(clonedStudentAnswerSelect);

    let subquestionTexts = document.getElementsByClassName("subquestion-text");
    let subquestionText = subquestionTexts[subquestionTexts.length - 1];
    let clonedSubquestionText = subquestionText.cloneNode(true);
    clonedSubquestionText.value = "";
    additionalQuestions.appendChild(clonedSubquestionText);
}

/**
 * Shuffles array
 * Used to randomize the order of possible answers presented to the user (they may be ordered the same as correct answers)
 * @param {object} array Array to be shuffled
 */
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        // Generate random number
        let j = Math.floor(Math.random() * (i + 1));
        let temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}

/**
 * Automatic update of student's available answers when dropdown menus are used after student selects an answer
 * Subquestion types - 1, 3, 9
 */
function updateStudentsAnswersSelect() {
    let possibleAnswerArray = [];
    $('.possible-answer-text').each(function () {
        let answer = $(this).text();
        possibleAnswerArray.push(answer);
    });
    let studentAnswerArray = [];
    $('select.student-answer-select').each(function () {
        let answer = $(this).val();
        studentAnswerArray.push(answer);
    });
    let availableStudentAnswerArray = possibleAnswerArray.filter((item) => !studentAnswerArray.includes(item));
    //clear all existing student's answers
    $('select.student-answer-select').each(function () {
        $(this).empty();
    });
    let studentAnswerSelect = document.getElementsByClassName('student-answer-select');
    for (let i = 0; i < studentAnswerSelect.length; i++) {
        //add currently selected option to each element
        let opt = document.createElement('option');
        opt.value = studentAnswerArray[i];
        opt.innerHTML = studentAnswerArray[i];
        studentAnswerSelect.item(i).appendChild(opt);

        //add remaining available options to each element
        for (let j = 0; j < availableStudentAnswerArray.length; j++) {
            let opt = document.createElement('option');
            opt.value = availableStudentAnswerArray[j];
            opt.innerHTML = availableStudentAnswerArray[j];
            studentAnswerSelect.item(i).appendChild(opt);
        }
    }
}

/**
 * Called after student's answer is submitted on the SolveQuestion page
 * Modifies certain fields so that they are properly bounded to the SubquestionResult
 * @param {number} subquestionType Type of the subquestion
 */
function onSolveQuestionFormSubmission(subquestionType) {
    let table = document.getElementById("student-answers-table");
    let studentHiddenAnswers = document.getElementsByClassName("student-answer-hidden");
    switch (subquestionType) {
        case 2:
            for (let i = 1; i < table.rows.length; i++) {
                if (table.rows[i].cells[0].getElementsByTagName("input")[0].checked) {
                    studentHiddenAnswers.item(i - 1).name = "StudentsAnswers[]";
                    studentHiddenAnswers.item(i - 1).value = table.rows[i].cells[1].innerHTML;
                }
            }
            break;
        case 4:
            let possibleHiddenAnswers = document.getElementsByClassName("possible-answer-hidden");
            for (let i = 1; i < table.rows.length; i++) {
                if (table.rows[i].cells[1].getElementsByTagName("input")[0].checked) {
                    studentHiddenAnswers.item(i - 1).value = "1";
                }
                else if (table.rows[i].cells[2].getElementsByTagName("input")[0].checked) {
                    studentHiddenAnswers.item(i - 1).value = "0";
                }
                else {
                    studentHiddenAnswers.item(i - 1).value = "X";
                }

                possibleHiddenAnswers.item(i - 1).value = table.rows[i].cells[0].getElementsByTagName("div")[0].innerHTML;
            }
            break;
        case 6:
            for (let i = 1; i < table.rows.length; i++) {
                if (table.rows[i].cells[0].getElementsByTagName("input")[0].checked) {
                    studentHiddenAnswers.item(0).value = table.rows[i].cells[1].innerHTML;
                    break;
                }
            }
            break;
    }
}

/**
 * Used to reset student's answer to the subquestion
 * Useful in case student wants to change his answer
 * @param {number} subquestionType Type of the subquestion
 */
function resetStudentAnswers(subquestionType) {
    let studentAnswerTable = document.getElementById("student-answers-table");
    switch (subquestionType) {
        case 1:
        case 3:
        case 9:
            let possibleAnswerArray = [];
            possibleAnswerArray.push(getOptionPlaceholderText());
            $('.possible-answer-text').each(function () {
                let answer = $(this).text();
                possibleAnswerArray.push(answer);
            });

            let studentAnswerSelects = document.getElementsByClassName('student-answer-select');
            $(".student-answer-select").empty();
            for (let i = 0; i < studentAnswerSelects.length; i++) {
                for (let j = 0; j < possibleAnswerArray.length; j++) {
                    let opt = document.createElement('option');
                    opt.value = possibleAnswerArray[j];
                    opt.innerHTML = possibleAnswerArray[j];
                    studentAnswerSelects.item(i).appendChild(opt);
                }
            }
            break;
        case 2:
        case 6:
            for (let i = 1; i < studentAnswerTable.rows.length; i++) {
                studentAnswerTable.rows[i].cells[0].getElementsByTagName("input")[0].checked = false;
            }
            break;
        case 4:
            for (let i = 1; i < studentAnswerTable.rows.length; i++) {
                studentAnswerTable.rows[i].cells[1].getElementsByTagName("input")[0].checked = false;
                studentAnswerTable.rows[i].cells[2].getElementsByTagName("input")[0].checked = false;
            }
            break;
        case 5:
            document.getElementById("student-answer").value = "";
            break;
        case 7:
            let studentAnswerSelect = document.getElementById('student-answer-select');
            studentAnswerSelect.value = getOptionPlaceholderText();
            break;
        case 8:
            document.getElementById("gap-text").value = "";
            break;
        case 10:
            let slider = document.getElementById("slider-question");
            slider.value = Math.round((parseInt(slider.min) + parseInt(slider.max)) / 2);
            changeSliderOutputs("Nezodpovězeno");
            break;
    }
}

/**
 * Called after student moves the slider (or removes his previous answer)
 * Used to change the value of the label that indicates the current value of the slider
 * @param {string} value New value of the slider
 */
function changeSliderOutputs(value) {
    document.getElementById("slider-question").parentNode.nextElementSibling.value = value;
    document.getElementById("student-answer-hidden").value = value;
}

/**
 * Called after SolveQuestion page is loaded
 * Used to add navigation table elements that the user can use to navigate throughout the test
 * @param {string} answerCompletenessString List of completeness of all student's answers (whether the student has answered the subquestions or not), separated by ";"
 * @param {number} subquestionNumber Index of current subquestion
 */
function addTestNavigationTableElements(answerCompletenessString, subquestionNumber) {
    let navigation = null;
    if (document.getElementById("number-list-navigation") == null) {
        navigation = document.getElementById("solved-number-list-navigation");
    }
    else {
        navigation = document.getElementById("number-list-navigation");
    }
    let navigationElements = document.getElementsByClassName("navigation-element");
    let navigationElement = navigationElements[0];

    let answerCompletenessArray = [];
    let answerCompletenessStringSplit = answerCompletenessString.split(";");
    for (let i = 0; i < answerCompletenessStringSplit.length - 1; i++) {
        answerCompletenessArray.push(answerCompletenessStringSplit[i]);
    }
    let colorArray = ["gray", "lightgreen", "orange", "red", "gray", "gray"];
    navigationElement.style.backgroundColor = colorArray[answerCompletenessArray[0]];//set color of first navigation element manually

    for (let i = 1; i < answerCompletenessArray.length; i++) {
        let clonedNavigationElement = navigationElement.cloneNode(true);
        clonedNavigationElement.value = (i + 1);
        clonedNavigationElement.innerHTML = (i + 1);
        clonedNavigationElement.style.backgroundColor = colorArray[answerCompletenessArray[i]];
        if (subquestionNumber == i) {
            clonedNavigationElement.style.fontWeight = "1000"
        }
        navigation.appendChild(clonedNavigationElement);
    }

    if (subquestionNumber == 0) {
        navigationElement.style.fontWeight = "1000"
    }
}

/**
 * Used by student to navigate throughout the test attempt
 * @param {number} subquestionIndex Index of the subquestion to navigate onto
 */
function navigateToSubquestion(subquestionIndex) {
    subquestionIndex -= 1;
    let nextSubquestionButton = document.getElementById("nextSubquestion");
    nextSubquestionButton.disabled = false;
    nextSubquestionButton.value = subquestionIndex;
    nextSubquestionButton.click();
}

/**
 * Used by student to turn test in
 */
function turnTestIn() {
    if (confirm('Chystáte se odevzdat test. V testu již nebude možné provádět žádné změny. Chcete pokračovat?')) {
        let action = document.getElementById("action");
        action.value = "turnTestIn";
        let nextSubquestionButton = document.getElementById("nextSubquestion");
        nextSubquestionButton.click();
    }
}

//AddTestTemplate.cshtml / EditTestTemplate.cshtml

/**
 * Called after the AddTestTemplate/EditTestTemplate page is loaded
 * Used to change certain fields (negative points radiobutton and subject dropdown)
 * @param {number} negativePoints Negative points settings (1/2/3)
 * @param {number} subjectId ID of the test template subject
 */
function editTestTemplatePostProcessing(negativePoints, subjectId) {
    document.getElementById(negativePoints).checked = true;
    document.getElementById("subject").value = subjectId;
}

/**
 * Called after lecturer requests test difficulty comparison
 * Hides button used to request comparison and lets the lecturer know that the calculation is being performed
 */
function onTestDifficultyFormSubmission() {
    let suggestedPointsLabel = document.getElementById("suggested-points-label");
    suggestedPointsLabel.innerHTML = "Předpokládaný průměrný počet bodů: probíhá výpočet..";
    let suggestedPointsButton = document.getElementById("suggested-points-button");
    suggestedPointsButton.style.display = "none";
}


//SolvedQuestion.cshtml

/**
 * Called after the SolvedQuestion page is loaded
 * Disables certain fields and adds navigation elements
 * @param {number} subquestionNumber Index of current subquestion
 * @param {number} subquestionsCount Amount of subquestion in the test
 * @param {string} answerStatusString List of correctness of all student's answers (whether the student has answered the subquestions correctly or not), separated by ";"
 */
function solvedQuestionPagePostProcessing(subquestionNumber, subquestionsCount, answerStatusString) {
    if (subquestionNumber == 0 || subquestionsCount == 1) {
        document.getElementById("previousSubquestion").disabled = true;
    }
    if ((subquestionNumber == subquestionsCount - 1) || subquestionsCount <= 1) {
        document.getElementById("nextSubquestion").disabled = true;
    }
    addTestNavigationTableElements(answerStatusString, subquestionNumber);
}

/**
 * Used by student to navigate throughout the test result
 * @param {number} subquestionIndex Index of the subquestion to navigate onto
 */
function navigateToSolvedSubquestion(subquestionIndex) {
    subquestionIndex -= 2;
    let subquestionResultIndex = document.getElementById("subquestionResultIndex");
    subquestionResultIndex.value = subquestionIndex;
    let nextSubquestionButton = document.getElementById("nextSubquestion");
    nextSubquestionButton.disabled = false;
    nextSubquestionButton.click();
}

//ManageSolvedQuestion.cshtml

/**
 * Called after lecturer requests subquestion result suggestion
 * Hides button used to request suggestion and lets the lecturer know that the calculation is being performed
 */
function onSubquestionResultPointsFormSubmission() {
    let suggestedPointsLabel = document.getElementById("suggested-points-label");
    suggestedPointsLabel.innerHTML = "Doporučený počet bodů za otázku: probíhá výpočet..";
    let suggestedPointsButton = document.getElementById("suggested-points-button");
    suggestedPointsButton.style.display = "none";
}

//ManageArtificialIntelligence.cshtml

/**
 * Lets the lecturer know that testing subquestion templates are being added
 */
function onAddTestingSubquestionTemplatesFormSubmission() {
    let templatesAddedLabel = document.getElementById("templates-added-label");
    templatesAddedLabel.style.display = "inline";
}

/**
 * Lets the lecturer know that testing subquestion results are being added
 */
function onAddTestingSubquestionResultsFormSubmission() {
    let templatesAddedLabel = document.getElementById("results-added-label");
    templatesAddedLabel.style.display = "inline";
}

//General

/**
 * Used by the lecturer to hide confirm action form (lecturer has chosen not to perform a potentially dangerous action)
 */
function hideConfirmActionForm() {
    document.getElementById("confirm-action").style.display = "none";
}

/**
 * Loads modal image that is shown to the user in case he clicks on the default image
 */
function addModalImage() {
    let modal = document.getElementById("myModal");

    let img = document.getElementById("image");
    let modalImg = document.getElementById("modalimage");
    img.onclick = function () {
        modal.style.display = "block";
        modalImg.src = document.getElementById("hiddenimage").src;
    }

    let span = document.getElementsByClassName("close")[0];

    span.onclick = function () {
        modal.style.display = "none";
    }
}

/**
 * cs-CZ culture is used in the application - it's necessary to replace all dots with commas
 */
function modifyInputNumbers() {
    let inputs = document.getElementsByClassName("input-number");
    for (let i = 0; i < inputs.length; i++) {
        inputs[i].addEventListener('input', function () {
            this.value = this.value.replace(/[^0-9-,]/, '');
        });
    }
}