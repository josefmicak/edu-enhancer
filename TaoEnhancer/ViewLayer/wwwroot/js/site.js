// Please see documentation at https://docs.microsoft.com/aspnet/core/client-side/bundling-and-minification
// for details on configuring this project to bundle and minify static web assets.

// Write your JavaScript code.

function manageTest(testNameIdentifier, testNumberIdentifier) {
    window.alert("sometext");
}

function loadQuestionInfo(itemNumberIdentifier, itemNameIdentifier, title, label, points) {
    document.getElementById("testtemplate-item-itemnumberidentifier").innerHTML = "Číselný identifikátor otázky: " + itemNumberIdentifier;
    document.getElementById("testtemplate-item-itemnameidentifier").innerHTML = "Jmenný identifikátor otázky: " + itemNameIdentifier;
    document.getElementById("testtemplate-item-title").innerHTML = "Nadpis otázky: " + title;
    document.getElementById("testtemplate-item-label").innerHTML = "Označení otázky: " + label;
    document.getElementById("testtemplate-item-points").innerHTML = "Počet bodů za otázku: " + points;
}

function bla() {
    document.getElementById("testtemplate-negative-points-no").checked = true;
}

