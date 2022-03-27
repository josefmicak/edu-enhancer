using Common;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using ViewLayer.Models;

namespace ViewLayer.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private ItemController itemController = new ItemController();
        private StudentController studentController = new StudentController();
        private TestController testController = new TestController();

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View(new IndexModel {
                Title = "Přihlášení",
                SignInURL = Settings.GetSignInURL()
            });
        }

        public IActionResult TeacherMenu()
        {
            return View(new PageModel {
                Title = "Učitelské menu"
            });
        }

        public IActionResult StudentMenu()
        {
            return View(new StudentMenuModel {
                Title = "Studentské menu",
                Students = studentController.LoadStudents()
            });
        }

        public IActionResult TestTemplateList()
        {
            return View(new TestTemplateListModel {
                Title = "Správa zadání testů",
                Tests = testController.LoadTests()
            });
        }

        public IActionResult ManageSolvedTestList()
        {
            return View(new ManageSolvedTestListModel {
                Title = "Správa vyřešených testů",
                SolvedTests = testController.LoadSolvedTests()
            });
        }

        public IActionResult ManageSolvedTest(string testNameIdentifier, string testNumberIdentifier, string deliveryExecutionIdentifier, string studentIdentifier)
        {
            List<(string, string, string, string, int, bool)> itemParameters = testController.LoadItemInfo(testNameIdentifier, testNumberIdentifier);
            (List<(double questionResultPoints, bool questionResultPointsDetermined)> studentsPoints, int errorMessageNumber) questionResultPoints = testController.GetQuestionResultPoints(itemParameters, testNameIdentifier, testNumberIdentifier, deliveryExecutionIdentifier);

            return View(new ManageSolvedTestModel
            {
                Title = "Správa vyřešeného testu " + deliveryExecutionIdentifier,
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                DeliveryExecutionIdentifier = deliveryExecutionIdentifier,
                StudentIdentifier = studentIdentifier,
                TestParameters = testController.LoadTestParameters(testNameIdentifier, testNumberIdentifier),
                QuestionList = testController.LoadQuestions(testNameIdentifier, testNumberIdentifier),
                ItemParameters = itemParameters,
                TestPoints = testController.GetTestPoints(itemParameters),
                QuestionResultPoints = questionResultPoints,
                TotalStudentsPoints = testController.GetTotalStudentsPoints(questionResultPoints.studentsPoints),
                ResultParameters = testController.LoadResultParameters(testNameIdentifier, deliveryExecutionIdentifier, studentIdentifier)
            });
        }

        [HttpGet]
        public IActionResult TestTemplate(string testNameIdentifier, string testNumberIdentifier)
        {
            List<(string, string, string, string, int, bool)> itemParameters = testController.LoadItemInfo(testNameIdentifier, testNumberIdentifier);
            
            return View(new TestTemplateModel
            {
                Title = "Správa zadání testu " + testNameIdentifier,
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                TestParameters = testController.LoadTestParameters(testNameIdentifier, testNumberIdentifier),
                QuestionList = testController.LoadQuestions(testNameIdentifier, testNumberIdentifier),
                ItemParameters = itemParameters,
                TestPoints = testController.GetTestPoints(itemParameters),
                NegativePoints = testController.NegativePointsManagement(testNameIdentifier, testNumberIdentifier)
            });
        }

        [HttpPost]
        public IActionResult TestTemplate(string testNameIdentifier, string testNumberIdentifier, string negativePoints)
        {
            string textToWrite = "";
            if (negativePoints == "negativePoints_no")
            {
                textToWrite = "0";
            }
            else if (negativePoints == "negativePoints_yes")
            {
                textToWrite = "1";
            }
            System.IO.File.WriteAllText(Settings.GetTestTestNegativePointsDataPath(testNameIdentifier, testNumberIdentifier), textToWrite);
            
            List<(string, string, string, string, int, bool)> itemParameters = testController.LoadItemInfo(testNameIdentifier, testNumberIdentifier);

            return View(new TestTemplateModel {
                Title = "Správa zadání testu " + testNameIdentifier,
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                TestParameters = testController.LoadTestParameters(testNameIdentifier, testNumberIdentifier),
                QuestionList = testController.LoadQuestions(testNameIdentifier, testNumberIdentifier),
                ItemParameters = itemParameters,
                TestPoints = testController.GetTestPoints(itemParameters),
                NegativePoints = testController.NegativePointsManagement(testNameIdentifier, testNumberIdentifier),
                NegativePointsOption = negativePoints
            });
        }

        [HttpGet]
        public IActionResult ItemTemplate(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier)
        {
            (string, string, string title, string label, int amountOfSubitems) itemParameters = itemController.LoadItemParameters(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier);
            (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) responseIdentifiers = itemController.GetResponseIdentifiers(itemParameters.amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
            string responseIdentifier = responseIdentifiers.responseIdentifierArray[0];
            (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) subitemParameters = itemController.LoadSubitemParameters(responseIdentifier, itemParameters.amountOfSubitems, responseIdentifiers.responseIdentifierArray, responseIdentifiers.responseValueArray, testNameIdentifier, itemNumberIdentifier);
            double correctChoicePoints = itemController.GetCorrectChoicePoints(subitemParameters.subquestionPoints, subitemParameters.correctChoiceArray, subitemParameters.questionType);
            (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) questionPoints = itemController.LoadQuestionPoints(testNameIdentifier, itemNumberIdentifier, responseIdentifier, itemParameters.amountOfSubitems, correctChoicePoints);

            return View(new ItemTemplateModel
            {
                Title = "Správa zadání otázky " + itemNumberIdentifier + " / " + itemNameIdentifier,
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                ItemNameIdentifier = itemNameIdentifier,
                ItemNumberIdentifier = itemNumberIdentifier,
                ItemParameters = itemParameters,
                ResponseIdentifiers = responseIdentifiers,
                ResponseIdentifier = responseIdentifier,
                SubitemParameters = subitemParameters,
                QuestionPoints = questionPoints,
                QuestionTypeText = itemController.GetQuestionTypeText(subitemParameters.questionType),
                IsSelectDisabled = (itemParameters.amountOfSubitems > 1 ? false : true),
                CorrectChoicePoints = correctChoicePoints,
                CorrectChoiceArray = subitemParameters.correctChoiceArray,
                CorrectAnswerCount = (subitemParameters.questionType == 3 || subitemParameters.questionType == 4 ? subitemParameters.correctAnswerArray.Count / 2 : subitemParameters.correctAnswerArray.Count),
                WrongChoicePoints = subitemParameters.wrongChoicePoints,
                SubquestionPoints = subitemParameters.subquestionPoints
            });
        }

        [HttpPost]
        public IActionResult ItemTemplate(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier, string selectedSubitem, string subquestionPoints,
            string wrongChoicePoints, string recommendedWrongChoicePoints, string selectedWrongChoicePoints, int correctChoicePoints, List<string> correctChoiceArray, int questionType)
        {
            string errorText = "";
            if (subquestionPoints != null)
            {
                bool isNumber = int.TryParse(subquestionPoints, out _);
                if (!isNumber)
                {
                    errorText = "Chyba: je nutné zadat číslo.";
                }
                else
                {
                    if (wrongChoicePoints == "wrongChoicePoints_recommended")
                    {
                        double recommendedWrongChoicePointsRecounted = itemController.GetCorrectChoicePoints(int.Parse(subquestionPoints), correctChoiceArray, questionType) * (-1);
                        wrongChoicePoints = recommendedWrongChoicePointsRecounted.ToString();
                    }
                    else
                    {
                        wrongChoicePoints = selectedWrongChoicePoints;
                    }

                    if (recommendedWrongChoicePoints == "N/A")
                    {
                        wrongChoicePoints = (correctChoicePoints * (-1)).ToString();
                    }

                    bool isWrongChoicePointsNumber = double.TryParse(wrongChoicePoints, out _);

                    if (!isWrongChoicePointsNumber)
                    {
                        errorText = "Chyba: je nutné zadat číslo.";
                    }
                    else
                    {
                        bool performSave = true;
                        int subquestionPointsToSave = int.Parse(subquestionPoints);
                        double wrongChoicePointsToSave = double.Parse(wrongChoicePoints);

                        if (Math.Abs(wrongChoicePointsToSave) > subquestionPointsToSave)
                        {
                            performSave = false;
                            errorText = "Chyba: za špatný výběr bude studentovi odečteno více bodů, než kolik může dostat za otázku.";
                        }

                        if (wrongChoicePointsToSave > 0)
                        {
                            performSave = false;
                            errorText = "Chyba: za špatnou volbu nemůže být udělen kladný počet bodů.";
                        }

                        if (subquestionPointsToSave < 0)
                        {
                            performSave = false;
                            errorText = "Chyba: za správnou odpověď nemůže být udělen záporný počet bodů.";
                        }

                        if (performSave)
                        {
                            string[] importedFileLines = System.IO.File.ReadAllLines(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier));
                            string fileLinesToExport = "";
                            for (int i = 0; i < importedFileLines.Length; i++)
                            {
                                string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                                if (splitImportedFileLineBySemicolon[0] == selectedSubitem)
                                {
                                    importedFileLines[i] = selectedSubitem + ";" + subquestionPointsToSave + ";" + Math.Round(wrongChoicePointsToSave, 2);
                                }
                                fileLinesToExport += importedFileLines[i] + "\n";
                            }
                            System.IO.File.WriteAllText(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier), fileLinesToExport);
                        }
                    }
                }
            }
            (string, string, string title, string label, int amountOfSubitems) itemParameters = itemController.LoadItemParameters(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier);
            (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) responseIdentifiers = itemController.GetResponseIdentifiers(itemParameters.amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
            string responseIdentifier = (itemParameters.amountOfSubitems == 1 || selectedSubitem == null ? responseIdentifiers.responseIdentifierArray[0] : selectedSubitem);
            (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) subitemParameters = itemController.LoadSubitemParameters(responseIdentifier, itemParameters.amountOfSubitems, responseIdentifiers.responseIdentifierArray, responseIdentifiers.responseValueArray, testNameIdentifier, itemNumberIdentifier);
            (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) questionPoints = itemController.LoadQuestionPoints(testNameIdentifier, itemNumberIdentifier, responseIdentifier, itemParameters.amountOfSubitems, correctChoicePoints);

            ItemTemplateModel model = new ItemTemplateModel()
            {
                Title = "Správa zadání otázky " + itemNumberIdentifier + " / " + itemNameIdentifier,
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                ItemNameIdentifier = itemNameIdentifier,
                ItemNumberIdentifier = itemNumberIdentifier
            };
            model.ItemParameters = itemParameters;
            model.ResponseIdentifiers = responseIdentifiers;
            model.ResponseIdentifier = responseIdentifier;
            model.SubitemParameters = subitemParameters;
            model.QuestionPoints = questionPoints;
            model.QuestionTypeText = itemController.GetQuestionTypeText(subitemParameters.questionType);
            model.IsSelectDisabled = (itemParameters.amountOfSubitems > 1 ? false : true);
            model.CorrectChoicePoints = (subquestionPoints != null ? itemController.GetCorrectChoicePoints(int.Parse(subquestionPoints), correctChoiceArray, questionType) : subitemParameters.subquestionPoints);
            model.CorrectChoiceArray = correctChoiceArray;
            model.CorrectAnswerCount = (subitemParameters.questionType == 3 || subitemParameters.questionType == 4 ? subitemParameters.correctAnswerArray.Count / 2 : subitemParameters.correctAnswerArray.Count);
            if (wrongChoicePoints != null) { model.WrongChoicePoints = double.Parse(wrongChoicePoints); }
            else { model.WrongChoicePoints = subitemParameters.wrongChoicePoints; }
            if (subquestionPoints != null) { model.SubquestionPoints = int.Parse(subquestionPoints); }
            else { model.SubquestionPoints = subitemParameters.subquestionPoints; }
            model.SubquestionPointsText = subquestionPoints;
            model.ErrorText = errorText;

            return View(model);
        }

        [HttpGet]
        public IActionResult ManageSolvedItem(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier, string deliveryExecutionIdentifier, string studentIdentifier)
        {
            (string, string, string title, string label, int amountOfSubitems) itemParameters = itemController.LoadItemParameters(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier);
            (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) responseIdentifiers = itemController.GetResponseIdentifiers(itemParameters.amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
            string responseIdentifier = responseIdentifiers.responseIdentifierArray[0];
            (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) subitemParameters = itemController.LoadSubitemParameters(responseIdentifier, itemParameters.amountOfSubitems, responseIdentifiers.responseIdentifierArray, responseIdentifiers.responseValueArray, testNameIdentifier, itemNumberIdentifier);
            List<double> studentsSubitemPointsList = itemController.GetStudentsSubitemPointsList(testNameIdentifier, itemNameIdentifier, deliveryExecutionIdentifier);
            double correctChoicePoints = itemController.GetCorrectChoicePoints(subitemParameters.subquestionPoints, subitemParameters.correctChoiceArray, subitemParameters.questionType);
            (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) questionPoints = itemController.LoadQuestionPoints(testNameIdentifier, itemNumberIdentifier, responseIdentifier, itemParameters.amountOfSubitems, correctChoicePoints);
            (double, List<string> studentsAnswers, string studentsAnswerCorrectLabel, string studentsAnswerPointsLabel) deliveryExecutionInfo = itemController.LoadDeliveryExecutionInfo(testNameIdentifier, testNumberIdentifier, itemNumberIdentifier, itemNameIdentifier, responseIdentifier, deliveryExecutionIdentifier, subitemParameters.correctAnswerArray, subitemParameters.correctChoiceArray, subitemParameters.subquestionPoints, questionPoints.recommendedWrongChoicePoints, questionPoints.selectedWrongChoicePoints, true, itemController.GetCurrentSubitemIndex(responseIdentifier, responseIdentifiers.responseIdentifierArray));
            string answerClass = "";
            switch (deliveryExecutionInfo.studentsAnswerCorrectLabel)
            {
                case "Správná odpověď":
                    answerClass = "correct";
                    break;
                case "Částečně správná odpověď":
                    answerClass = "partiallyCorrect";
                    break;
                case "Nesprávná odpověď":
                    answerClass = "incorrect";
                    break;
            }

            return View(new ManageSolvedItemModel
            {
                Title = "Správa vyřešeného testu " + deliveryExecutionIdentifier + ", otázka " + itemNameIdentifier,
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                ItemNameIdentifier = itemNameIdentifier,
                ItemNumberIdentifier = itemNumberIdentifier,
                DeliveryExecutionIdentifier = deliveryExecutionIdentifier,
                StudentIdentifier = studentIdentifier,
                ItemParameters = itemParameters,
                ResponseIdentifiers = responseIdentifiers,
                ResponseIdentifier = responseIdentifier,
                SubitemParameters = subitemParameters,
                StudentsSubitemPointsList = studentsSubitemPointsList,
                StudentsSubitemPointsListSum = studentsSubitemPointsList.Sum(),
                StudentsSubitemPoints = itemController.GetStudentsSubitemPoints(studentsSubitemPointsList, responseIdentifier, responseIdentifiers.responseIdentifierArray),
                QuestionPoints = questionPoints,
                DeliveryExecutionInfo = deliveryExecutionInfo,
                AnswerClass = answerClass,
                QuestionTypeText = itemController.GetQuestionTypeText(subitemParameters.questionType),
                IsSelectDisabled = (itemParameters.amountOfSubitems > 1 ? false : true),
                CurrentSubitemIndex = itemController.GetCurrentSubitemIndex(responseIdentifier, responseIdentifiers.responseIdentifierArray)
            });
        }

        [HttpPost]
        public IActionResult ManageSolvedItem(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier, string deliveryExecutionIdentifier, string studentIdentifier, string selectedSubitem, string studentsPoints, int amountOfSubitems, int subitemIndex, int questionPointsDetermined)
        {
            string errorText = "";
            if (studentsPoints != null)
            {
                bool isDecimal = double.TryParse(studentsPoints, out _);
                if (!isDecimal)
                {
                    errorText = "Chyba: je nutné zadat číslo.";
                }
                else if (questionPointsDetermined == 0)
                {
                    errorText = "Chyba: není možné upravit počet bodů studenta. Nejprve je nutné určit počet obdržených bodů za otázku.";
                }
                else
                {
                    double studentsPointsToSave = double.Parse(studentsPoints);
                    string[] resultsFileLines = System.IO.File.ReadAllLines(Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier));
                    string resultsToFile = "";

                    for (int i = 0; i < resultsFileLines.Length; i++)
                    {
                        string[] splitResultsFileLineBySemicolon = resultsFileLines[i].Split(";");
                        if (splitResultsFileLineBySemicolon[0] != itemNameIdentifier)
                        {
                            resultsToFile += resultsFileLines[i] + "\n";
                        }
                        else
                        {
                            if (amountOfSubitems > 1)
                            {
                                resultsToFile += itemNameIdentifier;

                                for (int j = 1; j < splitResultsFileLineBySemicolon.Length; j++)
                                {
                                    resultsToFile += ";";
                                    if (j - 1 != subitemIndex)
                                    {
                                        resultsToFile += splitResultsFileLineBySemicolon[j];
                                    }
                                    else
                                    {
                                        resultsToFile += studentsPoints;
                                    }
                                }

                                resultsToFile += "\n";
                            }
                            else
                            {
                                resultsToFile += itemNameIdentifier + ";" + studentsPoints + "\n";
                            }
                        }
                    }

                    System.IO.File.WriteAllText(Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier), resultsToFile);
                }
            }
            (string, string, string title, string label, int amountOfSubitems) itemParameters = itemController.LoadItemParameters(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier);
            (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) responseIdentifiers = itemController.GetResponseIdentifiers(itemParameters.amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
            string responseIdentifier = (itemParameters.amountOfSubitems == 1 || selectedSubitem == null ? responseIdentifiers.responseIdentifierArray[0] : selectedSubitem);
            (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) subitemParameters = itemController.LoadSubitemParameters(responseIdentifier, itemParameters.amountOfSubitems, responseIdentifiers.responseIdentifierArray, responseIdentifiers.responseValueArray, testNameIdentifier, itemNumberIdentifier);
            List<double> studentsSubitemPointsList = itemController.GetStudentsSubitemPointsList(testNameIdentifier, itemNameIdentifier, deliveryExecutionIdentifier);
            double correctChoicePoints = itemController.GetCorrectChoicePoints(subitemParameters.subquestionPoints, subitemParameters.correctChoiceArray, subitemParameters.questionType);
            (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) questionPoints = itemController.LoadQuestionPoints(testNameIdentifier, itemNumberIdentifier, responseIdentifier, itemParameters.amountOfSubitems, correctChoicePoints);
            (double, List<string> studentsAnswers, string studentsAnswerCorrectLabel, string studentsAnswerPointsLabel) deliveryExecutionInfo = itemController.LoadDeliveryExecutionInfo(testNameIdentifier, testNumberIdentifier, itemNumberIdentifier, itemNameIdentifier, responseIdentifier, deliveryExecutionIdentifier, subitemParameters.correctAnswerArray, subitemParameters.correctChoiceArray, subitemParameters.subquestionPoints, questionPoints.recommendedWrongChoicePoints, questionPoints.selectedWrongChoicePoints, true, itemController.GetCurrentSubitemIndex(responseIdentifier, responseIdentifiers.responseIdentifierArray));
            string answerClass = "";
            switch (deliveryExecutionInfo.studentsAnswerCorrectLabel)
            {
                case "Správná odpověď":
                    answerClass = "correct";
                    break;
                case "Částečně správná odpověď":
                    answerClass = "partiallyCorrect";
                    break;
                case "Nesprávná odpověď":
                    answerClass = "incorrect";
                    break;
            }

            return View(new ManageSolvedItemModel {
                Title = "Správa vyřešeného testu " + deliveryExecutionIdentifier + ", otázka " + itemNameIdentifier,
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                ItemNameIdentifier = itemNameIdentifier,
                ItemNumberIdentifier = itemNumberIdentifier,
                DeliveryExecutionIdentifier = deliveryExecutionIdentifier,
                StudentIdentifier = studentIdentifier,
                ItemParameters = itemParameters,
                ResponseIdentifiers = responseIdentifiers,
                ResponseIdentifier = responseIdentifier,
                SubitemParameters = subitemParameters,
                StudentsSubitemPointsList = studentsSubitemPointsList,
                StudentsSubitemPointsListSum = studentsSubitemPointsList.Sum(),
                StudentsSubitemPoints = itemController.GetStudentsSubitemPoints(studentsSubitemPointsList, responseIdentifier, responseIdentifiers.responseIdentifierArray),
                QuestionPoints = questionPoints,
                DeliveryExecutionInfo = deliveryExecutionInfo,
                AnswerClass = answerClass,
                QuestionTypeText = itemController.GetQuestionTypeText(subitemParameters.questionType),
                IsSelectDisabled = (itemParameters.amountOfSubitems > 1 ? false : true),
                CurrentSubitemIndex = itemController.GetCurrentSubitemIndex(responseIdentifier, responseIdentifiers.responseIdentifierArray),
                StudentsPoints = studentsPoints,
                ErrorText = errorText
            });
        }

        public IActionResult BrowseSolvedTestList(string studentIdentifier)
        {
            return View(new BrowseSolvedTestListModel {
                Title = "Seznam testů studenta",
                StudentIdentifier = studentIdentifier,
                Student = studentController.LoadStudentByIdentifier(studentIdentifier),
                StudentTestList = testController.LoadTests(studentIdentifier)
            });
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

        public IActionResult BrowseSolvedTest(string studentIdentifier, string deliveryExecutionIdentifier, string testNameIdentifier, string testNumberIdentifier)
        {
            List<(string, string, string, string, int, bool)> itemParameters = testController.LoadItemInfo(testNameIdentifier, testNumberIdentifier);
            (List<(double questionResultPoints, bool questionResultPointsDetermined)> studentsPoints, int errorMessageNumber) questionResultPoints = testController.GetQuestionResultPoints(itemParameters, testNameIdentifier, testNumberIdentifier, deliveryExecutionIdentifier);

            return View(new BrowseSolvedTestModel {
                Title = "Prohlížení pokusu",
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                DeliveryExecutionIdentifier = deliveryExecutionIdentifier,
                StudentIdentifier = studentIdentifier,
                TestParameters = testController.LoadTestParameters(testNameIdentifier, testNumberIdentifier),
                QuestionList = testController.LoadQuestions(testNameIdentifier, testNumberIdentifier),
                ItemParameters = itemParameters,
                TestPoints = testController.GetTestPoints(itemParameters),
                QuestionResultPoints = questionResultPoints,
                TotalStudentsPoints = testController.GetTotalStudentsPoints(questionResultPoints.studentsPoints),
                ResultParameters = testController.LoadResultParameters(testNameIdentifier, deliveryExecutionIdentifier, studentIdentifier)
            });
        }

        [HttpGet]
        public IActionResult BrowseSolvedItem(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier, string deliveryExecutionIdentifier, string studentIdentifier)
        {
            (string, string, string title, string label, int amountOfSubitems) itemParameters = itemController.LoadItemParameters(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier);
            (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) responseIdentifiers = itemController.GetResponseIdentifiers(itemParameters.amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
            string responseIdentifier = responseIdentifiers.responseIdentifierArray[0];
            (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) subitemParameters = itemController.LoadSubitemParameters(responseIdentifier, itemParameters.amountOfSubitems, responseIdentifiers.responseIdentifierArray, responseIdentifiers.responseValueArray, testNameIdentifier, itemNumberIdentifier);
            List<double> studentsSubitemPointsList = itemController.GetStudentsSubitemPointsList(testNameIdentifier, itemNameIdentifier, deliveryExecutionIdentifier);
            double correctChoicePoints = itemController.GetCorrectChoicePoints(subitemParameters.subquestionPoints, subitemParameters.correctChoiceArray, subitemParameters.questionType);
            (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) questionPoints = itemController.LoadQuestionPoints(testNameIdentifier, itemNumberIdentifier, responseIdentifier, itemParameters.amountOfSubitems, correctChoicePoints);
            (double, List<string> studentsAnswers, string studentsAnswerCorrectLabel, string studentsAnswerPointsLabel) deliveryExecutionInfo = itemController.LoadDeliveryExecutionInfo(testNameIdentifier, testNumberIdentifier, itemNumberIdentifier, itemNameIdentifier, responseIdentifier, deliveryExecutionIdentifier, subitemParameters.correctAnswerArray, subitemParameters.correctChoiceArray, subitemParameters.subquestionPoints, questionPoints.recommendedWrongChoicePoints, questionPoints.selectedWrongChoicePoints, true, itemController.GetCurrentSubitemIndex(responseIdentifier, responseIdentifiers.responseIdentifierArray));
            string answerClass = "";
            switch (deliveryExecutionInfo.studentsAnswerCorrectLabel)
            {
                case "Správná odpověď":
                    answerClass = "correct";
                    break;
                case "Částečně správná odpověď":
                    answerClass = "partiallyCorrect";
                    break;
                case "Nesprávná odpověď":
                    answerClass = "incorrect";
                    break;
            }

            return View(new BrowseSolvedItemModel {
                Title = "Prohlížení vyřešeného testu",
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                ItemNameIdentifier = itemNameIdentifier,
                ItemNumberIdentifier = itemNumberIdentifier,
                DeliveryExecutionIdentifier = deliveryExecutionIdentifier,
                StudentIdentifier = studentIdentifier,
                ItemParameters = itemParameters,
                ResponseIdentifiers = responseIdentifiers,
                ResponseIdentifier = responseIdentifier,
                SubitemParameters = subitemParameters,
                StudentsSubitemPointsList = studentsSubitemPointsList,
                StudentsSubitemPointsListSum = studentsSubitemPointsList.Sum(),
                StudentsSubitemPoints = itemController.GetStudentsSubitemPoints(studentsSubitemPointsList, responseIdentifier, responseIdentifiers.responseIdentifierArray),
                QuestionPoints = questionPoints,
                DeliveryExecutionInfo = deliveryExecutionInfo,
                AnswerClass = answerClass,
                QuestionTypeText = itemController.GetQuestionTypeText(subitemParameters.questionType),
                IsSelectDisabled = (itemParameters.amountOfSubitems > 1 ? false : true)
            });
        }

        [HttpPost]
        public IActionResult BrowseSolvedItem(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier, string deliveryExecutionIdentifier, string studentIdentifier, string selectedSubitem)
        {
            (string, string, string title, string label, int amountOfSubitems) itemParameters = itemController.LoadItemParameters(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier);
            (List<string> responseIdentifierArray, List<string> responseValueArray, int errorMessageNumber) responseIdentifiers = itemController.GetResponseIdentifiers(itemParameters.amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
            string responseIdentifier = (itemParameters.amountOfSubitems == 1 || selectedSubitem == null ? responseIdentifiers.responseIdentifierArray[0] : selectedSubitem);
            (string responseIdentifierTemp, int questionType, int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints, string imageSource, string subitemText, List<string> possibleAnswerArray, List<string> subquestionArray, List<string> correctChoiceArray, List<string> correctAnswerArray) subitemParameters = itemController.LoadSubitemParameters(responseIdentifier, itemParameters.amountOfSubitems, responseIdentifiers.responseIdentifierArray, responseIdentifiers.responseValueArray, testNameIdentifier, itemNumberIdentifier);
            List<double> studentsSubitemPointsList = itemController.GetStudentsSubitemPointsList(testNameIdentifier, itemNameIdentifier, deliveryExecutionIdentifier);
            double correctChoicePoints = itemController.GetCorrectChoicePoints(subitemParameters.subquestionPoints, subitemParameters.correctChoiceArray, subitemParameters.questionType);
            (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) questionPoints = itemController.LoadQuestionPoints(testNameIdentifier, itemNumberIdentifier, responseIdentifier, itemParameters.amountOfSubitems, correctChoicePoints);
            (double, List<string> studentsAnswers, string studentsAnswerCorrectLabel, string studentsAnswerPointsLabel) deliveryExecutionInfo = itemController.LoadDeliveryExecutionInfo(testNameIdentifier, testNumberIdentifier, itemNumberIdentifier, itemNameIdentifier, responseIdentifier, deliveryExecutionIdentifier, subitemParameters.correctAnswerArray, subitemParameters.correctChoiceArray, subitemParameters.subquestionPoints, questionPoints.recommendedWrongChoicePoints, questionPoints.selectedWrongChoicePoints, true, itemController.GetCurrentSubitemIndex(responseIdentifier, responseIdentifiers.responseIdentifierArray));
            string answerClass = "";
            switch (deliveryExecutionInfo.studentsAnswerCorrectLabel)
            {
                case "Správná odpověď":
                    answerClass = "correct";
                    break;
                case "Částečně správná odpověď":
                    answerClass = "partiallyCorrect";
                    break;
                case "Nesprávná odpověď":
                    answerClass = "incorrect";
                    break;
            }

            return View(new BrowseSolvedItemModel
            {
                Title = "Prohlížení vyřešeného testu",
                TestNameIdentifier = testNameIdentifier,
                TestNumberIdentifier = testNumberIdentifier,
                ItemNameIdentifier = itemNameIdentifier,
                ItemNumberIdentifier = itemNumberIdentifier,
                DeliveryExecutionIdentifier = deliveryExecutionIdentifier,
                StudentIdentifier = studentIdentifier,
                ItemParameters = itemParameters,
                ResponseIdentifiers = responseIdentifiers,
                ResponseIdentifier = responseIdentifier,
                SubitemParameters = subitemParameters,
                StudentsSubitemPointsList = studentsSubitemPointsList,
                StudentsSubitemPointsListSum = studentsSubitemPointsList.Sum(),
                StudentsSubitemPoints = itemController.GetStudentsSubitemPoints(studentsSubitemPointsList, responseIdentifier, responseIdentifiers.responseIdentifierArray),
                QuestionPoints = questionPoints,
                DeliveryExecutionInfo = deliveryExecutionInfo,
                AnswerClass = answerClass,
                QuestionTypeText = itemController.GetQuestionTypeText(subitemParameters.questionType),
                IsSelectDisabled = (itemParameters.amountOfSubitems > 1 ? false : true)
            });
        }
    }
}