//ManageUserRegistrationList.cshtml

window.updateVisibility = function (accepted, rejected, text) {
    if (accepted == false) {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[6].textContent === "Schválena") {
                tr.style.display = 'none';
            }
        });
    }
    else {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[6].textContent === "Schválena") {
                tr.style.display = '';
            }
        });
    }

    if (rejected == false) {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[6].textContent === "Zamítnuta") {
                tr.style.display = 'none';
            }
        });
    }
    else {
        document.querySelectorAll('tr').forEach(tr => {
            if (tr.children[6].textContent === "Zamítnuta") {
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

    var fullName = table.rows[idArray[1]].cells[1].innerHTML;
    const nameArray = fullName.split(" "); 
    document.getElementById("studentFirstName").value = nameArray[0];
    document.getElementById("studentLastName").value = nameArray[1];

    var login = table.rows[idArray[1]].cells[2].innerHTML;
    document.getElementById("studentLogin").value = login;
}

function showEditStudentLabel(oldLogin, userIdentifier, email, firstName, lastName) {
    document.getElementById("student-action").value = 'editStudent';
    document.getElementById("added-student").style.visibility = 'hidden';
    document.getElementById("edited-student").style.visibility = 'visible';
    document.getElementById("studentOldLogin").value = oldLogin;
    document.getElementById("studentIdentifier").value = userIdentifier;
    document.getElementById("studentFirstName").value = firstName;
    document.getElementById("studentLastName").value = lastName;
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

function showEditTeacherLabel(oldLogin, email, firstName, lastName) {
    document.getElementById("teacher-action").value = 'editTeacher';
    document.getElementById("added-teacher").style.visibility = 'hidden';
    document.getElementById("edited-teacher").style.visibility = 'visible';
    document.getElementById("teacher-edit-role").style.visibility = 'visible';
    document.getElementById("teacherOldLogin").value = oldLogin;
    document.getElementById("teacherFirstName").value = firstName;
    document.getElementById("teacherLastName").value = lastName;
    document.getElementById("teacherLogin").value = oldLogin;
    document.getElementById('teacherLogin').readOnly = true;
    document.getElementById("teacherEmail").value = email;
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
    
    if (role == "3") {
        document.getElementById("isMainAdmin").value = false;
    }
    else if (role == "4") {
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

//show form which prompts user to confirm the action to the user

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

//Index.cshtml



//General

function hideConfirmActionForm() {
    document.getElementById("confirm-action").style.display = "none";
}