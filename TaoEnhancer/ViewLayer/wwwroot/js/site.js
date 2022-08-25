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

//ManageUserList.cshtml

function addStudentDetails(clicked_id) {
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