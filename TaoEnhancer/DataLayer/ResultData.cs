using Common;
using Common.Class;
using System.Xml;

namespace DataLayer
{
    public class ResultData
    {
        private StudentData studentData = new StudentData();
        private TestData testData = new TestData();

        /*----- Auxiliary functions -----*/
        private string ExtractFileName(string file)
        {
            string[] fileSplitBySlash = file.Split(@"\");
            return fileSplitBySlash[fileSplitBySlash.Length - 1];
        }

        private int GetAmountOfSubitems(string testNameIdentifier, string itemNumberIdentifier)
        {
            int amountOfSubitems = 0;
            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "responseDeclaration" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    amountOfSubitems++;
                }
            }
            return amountOfSubitems;
        }

        private double GetCorrectChoicePoints(int subquestionPoints, List<string> correctChoiceArray, int questionType)
        {
            double correctChoicePoints = 0;
            switch (questionType)
            {
                case int n when (n == 1 || n == 5 || n == 6 || n == 7 || n == 8 || n == 10):
                    correctChoicePoints = subquestionPoints;
                    break;
                case 2:
                    correctChoicePoints = (double)subquestionPoints / (double)correctChoiceArray.Count;
                    break;
                case int n when (n == 3 || n == 4 || n == 9):
                    correctChoicePoints = (double)subquestionPoints / (double)(correctChoiceArray.Count / 2);
                    break;
            }
            /* if (correctChoicePoints == Double.PositiveInfinity || correctChoicePoints == Double.NegativeInfinity)
             {
                 string errorMessage = "Chyba: otázka nemá pravděpodobně zadané žádné správné odpovědi.\nIdentifikátory otázky: " + @ViewBag.itemNameIdentifier + ", " + @ViewBag.itemNumberIdentifier;
                 WriteMessageToUser(errorMessage);
             }*/

            return Math.Round(correctChoicePoints, 2);
        }

        private double GetTotalStudentsPoints(List<(double, bool)> studentsPoints)
        {
            double totalStudentsPoints = 0;
            for (int i = 0; i < studentsPoints.Count; i++)
            {
                totalStudentsPoints += studentsPoints[i].Item1;
            }
            return totalStudentsPoints;
        }

        /*----- File functions -----*/
        /*private int CreateResultPointsFile(string testNameIdentifier, string deliveryExecutionIdentifier, List<(string, string, string, string, int, bool)> itemParameters)
        {
            string fileLinesToExport = "";
            int errorMessageNumber = 0;

            for (int i = 0; i < itemParameters.Count; i++)
            {
                string itemNumberIdentifier = itemParameters[i].Item1;
                string itemNameIdentifier = itemParameters[i].Item2;
                foreach (var directory in Directory.GetDirectories(Settings.GetTestItemsPath(testNameIdentifier)))
                {
                    if (itemNumberIdentifier == ExtractFileName(directory))
                    {
                        fileLinesToExport += itemNameIdentifier;
                        int amountOfSubitems = GetAmountOfSubitems(testNameIdentifier, itemNumberIdentifier);
                        (List<string> responseIdentifierArray, List<string> responseValueArray, errorMessageNumber) = GetResponseIdentifiers(amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
                        foreach (string responseIdentifier in responseIdentifierArray)
                        {
                            string responseIdentifierTemp = responseIdentifier;
                            string imageSource = "", subitemText = "";
                            bool subquestionPointsDetermined;
                            List<string> possibleAnswerArray = new List<string>();
                            List<string> subquestionArray = new List<string>();
                            List<string> correctChoiceArray = new List<string>();
                            List<string> correctAnswerArray = new List<string>();
                            int questionType = 0, subquestionPoints = 0;
                            double wrongChoicePoints = 0;
                            (responseIdentifierTemp, questionType, subquestionPoints, subquestionPointsDetermined, wrongChoicePoints, imageSource, subitemText, possibleAnswerArray, subquestionArray, correctChoiceArray, correctAnswerArray) = LoadSubitemParameters(responseIdentifier, amountOfSubitems, responseIdentifierArray, responseValueArray, testNameIdentifier, itemNumberIdentifier);
                            double correctChoicePoints = GetCorrectChoicePoints(subquestionPoints, correctChoiceArray, questionType);
                            (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) = LoadQuestionPoints(testNameIdentifier, itemNumberIdentifier, responseIdentifier, amountOfSubitems, correctChoicePoints);
                            (double studentsSubitemPoints, _, _, _) = LoadDeliveryExecutionInfo(testNameIdentifier, testNumberIdentifier, itemNumberIdentifier, itemNameIdentifier, responseIdentifier, deliveryExecutionIdentifier, correctAnswerArray, correctChoiceArray, subquestionPoints, recommendedWrongChoicePoints, selectedWrongChoicePoints, false, GetCurrentSubitemIndex(responseIdentifier, responseIdentifierArray));
                            fileLinesToExport += ";" + studentsSubitemPoints.ToString();
                        }
                        fileLinesToExport += "\n";
                    }
                }
            }
            string file = Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier);
            File.WriteAllText(file, fileLinesToExport);
            return errorMessageNumber;
        }*/

        private string[] LoadResults(string testNameIdentifier, string deliveryExecutionIdentifier)
        {
            string file = Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier);
            return File.ReadAllLines(file);
        }

        private void SaveResults(string testNameIdentifier, string itemNumberIdentifier, string fileLinesToExport)
        {
            string file = Settings.GetResultResultsDataPath(testNameIdentifier, itemNumberIdentifier);
            File.WriteAllText(file, fileLinesToExport);
        }

        // HomeController.cs
        private void UpdatePoints(string testNameIdentifier, string itemNumberIdentifier, string itemNameIdentifier, int amountOfSubitems, int subitemIndex, string studentsPoints)
        {
            string[] importedFileLines = LoadResults(testNameIdentifier, itemNumberIdentifier);
            string fileLinesToExport = "";
            for (int i = 0; i < importedFileLines.Length; i++)
            {
                string[] splitResultsFileLineBySemicolon = importedFileLines[i].Split(";");
                if (splitResultsFileLineBySemicolon[0] != itemNameIdentifier)
                {
                    fileLinesToExport += importedFileLines[i] + "\n";
                }
                else
                {
                    if (amountOfSubitems > 1)
                    {
                        fileLinesToExport += itemNameIdentifier;

                        for (int j = 1; j < splitResultsFileLineBySemicolon.Length; j++)
                        {
                            fileLinesToExport += ";";
                            if (j - 1 != subitemIndex)
                            {
                                fileLinesToExport += splitResultsFileLineBySemicolon[j];
                            }
                            else
                            {
                                fileLinesToExport += studentsPoints;
                            }
                        }

                        fileLinesToExport += "\n";
                    }
                    else
                    {
                        fileLinesToExport += itemNameIdentifier + ";" + studentsPoints + "\n";
                    }
                }
            }
            SaveResults(testNameIdentifier, itemNumberIdentifier, fileLinesToExport);
        }

        /*private (List<(double, bool)>, int) LoadQuestionResultPoints(List<(string, string, string, string, int, bool)> itemParameters, string testNameIdentifier, string deliveryExecutionIdentifier)
        {
            List<(double, bool)> studentsAnswerPoints = new List<(double, bool)>();
            int errorMessageNumber = 0;

            if(!File.Exists(Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier)))
            {
                errorMessageNumber = CreateResultPointsFile(testNameIdentifier, deliveryExecutionIdentifier, itemParameters);
            }

            for (int i = 0; i < itemParameters.Count; i++)
            {
                string itemNameIdentifier = itemParameters[i].Item2;
                double totalReceivedPoints = 0;
                string[] resultsFileLines = File.ReadAllLines(Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier));
                for (int j = 0; j < resultsFileLines.Length; j++)
                {
                    string[] splitResultsFileLineBySemicolon = resultsFileLines[j].Split(";");
                    if (splitResultsFileLineBySemicolon[0] == itemNameIdentifier)
                    {
                        for (int k = 1; k < splitResultsFileLineBySemicolon.Length; k++)
                        {
                            totalReceivedPoints += double.Parse(splitResultsFileLineBySemicolon[k]);
                        }
                        studentsAnswerPoints.Add((totalReceivedPoints, true));
                    }
                }
            }

            return (studentsAnswerPoints, errorMessageNumber);
        }*/

        private (string, string, string, string, string) LoadResultParameters(string testNameIdentifier, string deliveryExecutionIdentifier, string studentIdentifier)
        {
            string resultTimestamp = "";

            XmlReader xmlReader = XmlReader.Create(Settings.GetResultFilePath(testNameIdentifier, deliveryExecutionIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "testResult" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    resultTimestamp = xmlReader.GetAttribute("datestamp");
                }
            }

            Student student = studentData.Load(studentIdentifier);

            return (deliveryExecutionIdentifier, resultTimestamp, student.FirstName + " " + student.LastName, student.Login, studentIdentifier);
        }

        private List<double> GetStudentsSubitemPointsList(string testNameIdentifier, string itemNameIdentifier, string deliveryExecutionIdentifier)
        {
            List<double> studentsSubitemPoints = new List<double>();
            string[] resultsFileLines = LoadResults(testNameIdentifier, deliveryExecutionIdentifier);
            for (int i = 0; i < resultsFileLines.Length; i++)
            {
                string[] splitResultsFileLineBySemicolon = resultsFileLines[i].Split(";");
                if (splitResultsFileLineBySemicolon[0] == itemNameIdentifier)
                {
                    for (int j = 1; j < splitResultsFileLineBySemicolon.Length; j++)
                    {
                        studentsSubitemPoints.Add(double.Parse(splitResultsFileLineBySemicolon[j]));
                    }
                }
            }
            return studentsSubitemPoints;
        }

        /*private List<(string, string, string, string)> LoadSolvedStudentTests(string studentIdentifier)
        {
            List<(string, string, string, string)> studentTestList = new List<(string, string, string, string)>();

            foreach (var directory in Directory.GetDirectories(Settings.GetResultsPath()))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    string extension = ExtractFileExtension(file);
                    if (extension == "xml")
                    {
                        bool addTest = false;
                        string timeStamp = "";

                        XmlReader xmlReader = XmlReader.Create(file);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "context")
                            {
                                string testStudentIdentifier = xmlReader.GetAttribute("sourcedId");
                                if (testStudentIdentifier == studentIdentifier)
                                {
                                    addTest = true;
                                }
                            }

                            if (xmlReader.Name == "testResult" && xmlReader.GetAttribute("datestamp") != null)
                            {
                                timeStamp = xmlReader.GetAttribute("datestamp");
                            }
                        }

                        if (addTest)
                        {
                            string[] deliveryExecutionIdentifierSplitByUnderscore = ExtractFileNameWithoutExtension(file).Split("_");
                            string deliveryExecutionIdentifier = deliveryExecutionIdentifierSplitByUnderscore[2];
                            string testNameIdentifier = ExtractFileName(directory).ToString();
                            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);
                            studentTestList.Add((testNameIdentifier, timeStamp, deliveryExecutionIdentifier, testNumberIdentifier));
                        }
                    }
                }
            }

            return studentTestList;
        }*/

        /*private List<(string, string, string, string, string, string, string)> LoadSolvedTests()
        {
            List<(string, string, string, string, string, string, string)> solvedTestList = new List<(string, string, string, string, string, string, string)>();

            foreach (var directory in Directory.GetDirectories(Settings.GetResultsPath()))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    string extension = ExtractFileExtension(file);
                    if (extension == "xml")
                    {
                        string timeStamp = "";
                        string testStudentIdentifier = "";

                        XmlReader xmlReader = XmlReader.Create(file);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "context")
                            {
                                testStudentIdentifier = xmlReader.GetAttribute("sourcedId");
                            }

                            if (xmlReader.Name == "testResult" && xmlReader.GetAttribute("datestamp") != null)
                            {
                                timeStamp = xmlReader.GetAttribute("datestamp");
                            }
                        }

                        string[] attemptIdentifierSplitByUnderscore = ExtractFileNameWithoutExtension(file).Split("_");
                        string login = "", name = "", surname = "", email = "";

                        Student student = studentData.LoadStudent(testStudentIdentifier);

                        solvedTestList.Add((ExtractFileName(directory).ToString(), timeStamp, attemptIdentifierSplitByUnderscore[2], login, name + " " + surname, email, testStudentIdentifier));
                    }
                }
            }

            return solvedTestList;
        }*/

        //-------------------------------------------------- #OLD --------------------------------------------------

        /*// #TODO
        enum StudentsAnswerCorrectness
        {
            Correct,
            Incorrect,
            PartiallyCorrect,
            Unknown
        }

        // #TODO
        private void LoadDeliveryExecutionInfo(string testNameIdentifier, string resultNumberIdentifier, string itemNameIdentifier, string subitemIdentifier)
        {
            // #TODO
            List<string> correctChoiceArray = new List<string>();
            List<string> correctAnswerArray = new List<string>();
            List<(string, string)> choiceIdentifierValueTuple = new List<(string, string)>();
            int questionType = 0;
            int amountOfSubitems = 0;
            int subquestionPoints = 0;
            int questionPoints = 0;
            List<double> studentsReceivedPointsArray = new List<double>();
            bool deliveryExecutionFileCreated = false;
            List<double> importedReceivedPointsArray = new List<double>();
            double studentsReceivedPoints = 0;
            bool undecidedPointsInFile = false;
            bool negativePoints = false;
            bool recommendedWrongChoicePoints = true;
            double selectedWrongChoicePoints = 0;
            List<string> gapIdentifiers = new List<string>();

            //StudentsAnswerGB.Visible = true;
            List<string> studentsAnswers = new List<string>();

            foreach (var directory in Directory.GetDirectories(Settings.GetResultsPath()))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    string[] attemptIdentifierSplitByUnderscore = Path.GetFileNameWithoutExtension(file).Split("_");
                    if (attemptIdentifierSplitByUnderscore.Length > 2 && attemptIdentifierSplitByUnderscore[2] == resultNumberIdentifier)
                    {
                        XmlReader xmlReader = XmlReader.Create(file);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "itemResult")
                            {
                                if (xmlReader.GetAttribute("identifier") != itemNameIdentifier)
                                {
                                    xmlReader.Skip();
                                }
                            }

                            if (xmlReader.Name == "responseVariable")
                            {
                                if (xmlReader.GetAttribute("identifier") != subitemIdentifier)
                                {
                                    xmlReader.Skip();
                                }
                            }

                            if (xmlReader.Name == "outcomeVariable")
                            {
                                xmlReader.Skip();
                            }

                            if (xmlReader.Name == "value")
                            {
                                string studentsAnswer = xmlReader.ReadElementContentAsString();
                                if (questionType == 3 || questionType == 4 || questionType == 9)
                                {
                                    string[] studentsAnswerSplitBySpace = studentsAnswer.Split(" ");
                                    if (studentsAnswerSplitBySpace.Length == 2)
                                    {
                                        studentsAnswers.Add(studentsAnswerSplitBySpace[1]);
                                        studentsAnswers.Add(studentsAnswerSplitBySpace[0]);
                                    }
                                    else if (studentsAnswerSplitBySpace.Length == 3)
                                    {
                                        studentsAnswers.Add(studentsAnswerSplitBySpace[2]);
                                        studentsAnswers.Add(studentsAnswerSplitBySpace[1]);
                                    }
                                }
                                else if (questionType == 5 || questionType == 8)
                                {
                                    studentsAnswers.Add(studentsAnswer);
                                }
                                else
                                {
                                    string[] studentsAnswerSplitByApostrophe = studentsAnswer.Split("'");
                                    if (studentsAnswerSplitByApostrophe.Length > 1)
                                    {
                                        studentsAnswers.Add(studentsAnswerSplitByApostrophe[1]);
                                    }
                                    else
                                    {
                                        studentsAnswers.Add(studentsAnswer);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < studentsAnswers.Count; i++)
            {
                if (studentsAnswers[i] == "<>")
                {
                    string answer = studentsAnswers[i];
                    studentsAnswers.Remove(answer);
                }
            }

            string studentsAnswerToLabel = "";
            int answerNumber = 0;
            bool studentAnsweredQuestion = false;
            for (int i = 0; i < studentsAnswers.Count; i++)
            {
                for (int j = 0; j < choiceIdentifierValueTuple.Count; j++)
                {
                    if (studentsAnswers[i] == choiceIdentifierValueTuple[j].Item1)
                    {
                        if (questionType == 3 || questionType == 4)
                        {
                            if (answerNumber % 2 == 1)
                            {
                                studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + Environment.NewLine;
                            }
                            else
                            {
                                studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + " -> ";
                            }
                            answerNumber++;
                        }
                        else
                        {
                            studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + Environment.NewLine;
                        }
                    }
                }

                if (studentsAnswers[i] != "")
                {
                    studentAnsweredQuestion = true;
                }
            }

            if (questionType == 9)
            {
                studentsAnswerToLabel = "";
                int gapNumber = 0;

                for (int i = 0; i < gapIdentifiers.Count; i++)
                {
                    bool gapAnswered = false;
                    for (int j = 0; j < studentsAnswers.Count; j++)
                    {
                        if (j % 2 == 1)
                        {
                            continue;
                        }
                        if (studentsAnswers[j] == gapIdentifiers[i])
                        {
                            for (int k = 0; k < choiceIdentifierValueTuple.Count; k++)
                            {
                                if (studentsAnswers[j + 1] == choiceIdentifierValueTuple[k].Item1)
                                {
                                    gapAnswered = true;
                                    gapNumber++;
                                    studentsAnswerToLabel += "[" + gapNumber + "] - " + choiceIdentifierValueTuple[k].Item2 + Environment.NewLine;
                                }
                            }
                        }
                    }
                    if (!gapAnswered)
                    {
                        gapNumber++;
                        studentsAnswerToLabel += "[" + gapNumber + "] - nezodpovězeno" + Environment.NewLine;
                    }
                }
            }

            if (questionType == 5 || questionType == 8 || questionType == 10)
            {
                studentsAnswerToLabel = studentsAnswers[0];
            }

            if (!studentAnsweredQuestion)
            {
                studentsAnswerToLabel = "Nevyplněno";
            }

            //StudentsAnswerTB.Text = studentsAnswerToLabel;

            studentsReceivedPoints = 0;
            StudentsAnswerCorrectness isAnswerCorrect = StudentsAnswerCorrectness.Correct;

            if (deliveryExecutionFileCreated)
            {
                if (importedReceivedPointsArray.Count == 0)
                {
                    double totalReceivedPoints = 0;
                    string[] resultsFileLines = File.ReadAllLines(Settings.GetResultResultsDataPath(testNameIdentifier, resultNumberIdentifier));
                    for (int i = 0; i < resultsFileLines.Length; i++)
                    {
                        string[] splitResultsFileLineBySemicolon = resultsFileLines[i].Split(";");
                        if (splitResultsFileLineBySemicolon[0] == itemNameIdentifier)
                        {
                            for (int j = 1; j < splitResultsFileLineBySemicolon.Length; j++)
                            {
                                importedReceivedPointsArray.Add(double.Parse(splitResultsFileLineBySemicolon[j]));
                                totalReceivedPoints += double.Parse(splitResultsFileLineBySemicolon[j]);
                            }
                        }
                    }

                    //QuestionPointsLabel.Text = "Počet bodů za otázku: " + totalReceivedPoints + "/" + questionPoints.ToString();
                }

                if (amountOfSubitems > 1)
                {
                    // #TODO
                    studentsReceivedPoints = importedReceivedPointsArray[SubitemCB.SelectedIndex];
                }
                else
                {
                    studentsReceivedPoints = importedReceivedPointsArray[0];
                }
            }
            else
            {
                switch (questionType)
                {
                    case int n when (n == 1 || n == 6 || n == 7 || n == 10):
                        bool areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctChoiceArray, studentsAnswers);
                        if (areStudentsAnswersCorrect)
                        {
                            studentsReceivedPoints = subquestionPoints;
                        }
                        else
                        {
                            if (studentsAnswers.Count == 0 || (studentsAnswers.Count > 0 && studentsAnswers[0] != ""))
                            {
                                if (recommendedWrongChoicePoints)
                                {
                                    studentsReceivedPoints -= subquestionPoints;
                                }
                                else
                                {
                                    studentsReceivedPoints -= Math.Abs(selectedWrongChoicePoints);
                                }
                            }
                        }
                        break;
                    case 2:
                        int studentsCorrectAnswers = 0;

                        for (int i = 0; i < studentsAnswers.Count; i++)
                        {
                            for (int j = 0; j < correctChoiceArray.Count; j++)
                            {
                                if (studentsAnswers[i] == correctChoiceArray[j])
                                {
                                    studentsCorrectAnswers++;
                                    studentsReceivedPoints += ((double)subquestionPoints / (double)correctChoiceArray.Count);
                                }
                            }
                        }

                        if (recommendedWrongChoicePoints)
                        {
                            studentsReceivedPoints -= Math.Abs(studentsAnswers.Count - studentsCorrectAnswers) * ((double)subquestionPoints / (double)correctChoiceArray.Count);
                        }
                        else
                        {
                            studentsReceivedPoints -= Math.Abs(Math.Abs(studentsAnswers.Count - studentsCorrectAnswers) * (selectedWrongChoicePoints));
                        }
                        break;
                    case int n when (n == 3 || n == 4 || n == 9):
                        studentsCorrectAnswers = 0;

                        for (int i = 0; i < studentsAnswers.Count; i++)
                        {
                            for (int j = 0; j < correctChoiceArray.Count; j++)
                            {
                                if (i % 2 == 0 && j % 2 == 0)
                                {
                                    if ((studentsAnswers[i] == correctChoiceArray[j] && studentsAnswers[i + 1] == correctChoiceArray[j + 1]) ||
                                        (studentsAnswers[i + 1] == correctChoiceArray[j] && studentsAnswers[i] == correctChoiceArray[j + 1]))
                                    {
                                        studentsCorrectAnswers += 2;
                                        studentsReceivedPoints += ((double)subquestionPoints / (double)correctChoiceArray.Count) * 2;
                                    }
                                }
                            }
                        }

                        if (recommendedWrongChoicePoints)
                        {
                            studentsReceivedPoints -= Math.Abs(studentsAnswers.Count - studentsCorrectAnswers) * ((double)subquestionPoints / (double)correctChoiceArray.Count);
                        }
                        else
                        {
                            studentsReceivedPoints -= Math.Abs(Math.Abs((studentsAnswers.Count - studentsCorrectAnswers) / 2) * (selectedWrongChoicePoints));
                        }
                        break;
                    case 8:
                        if (correctAnswerArray[0] == studentsAnswers[0])
                        {
                            studentsReceivedPoints = subquestionPoints;
                        }
                        else
                        {
                            if (studentsAnswers.Count > 0 && studentsAnswers[0] != "")
                            {
                                if (recommendedWrongChoicePoints)
                                {
                                    studentsReceivedPoints -= subquestionPoints;
                                }
                                else
                                {
                                    studentsReceivedPoints -= Math.Abs(selectedWrongChoicePoints);
                                }
                            }
                        }
                        break;
                        case 9:
                            for(int i = 0; i < studentsAnswers.Count; i++)
                            {
                                Debug.WriteLine(i + ". studentsAnswers: " + studentsAnswers[i]);
                            }
                            for (int i = 0; i < correctChoiceArray.Count; i++)
                            {
                                Debug.WriteLine(i + ". correctChoiceArray: " + correctChoiceArray[i]);
                            }
                            break;
                }
            }

            switch (questionType)
            {
                case int n when (n == 1 || n == 6 || n == 7 || n == 10):
                    bool areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctChoiceArray, studentsAnswers);
                    if (!areStudentsAnswersCorrect)
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Incorrect;
                    }
                    else
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Correct;
                    }
                    break;
                case 2:
                    int studentsCorrectAnswers = 0;

                    for (int i = 0; i < studentsAnswers.Count; i++)
                    {
                        for (int j = 0; j < correctChoiceArray.Count; j++)
                        {
                            if (studentsAnswers[i] == correctChoiceArray[j])
                            {
                                studentsCorrectAnswers++;
                            }
                        }
                    }

                    if (studentsCorrectAnswers == 0)
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Incorrect;
                    }
                    else
                    {
                        if (studentsReceivedPoints != subquestionPoints || studentsAnswers.Count != correctChoiceArray.Count)
                        {
                            isAnswerCorrect = StudentsAnswerCorrectness.PartiallyCorrect;
                        }
                        else
                        {
                            isAnswerCorrect = StudentsAnswerCorrectness.Correct;
                        }
                    }
                    break;
                case int n when (n == 3 || n == 4 || n == 9):
                    studentsCorrectAnswers = 0;

                    for (int i = 0; i < studentsAnswers.Count; i++)
                    {
                        for (int j = 0; j < correctChoiceArray.Count; j++)
                        {
                            if (i % 2 == 0 && j % 2 == 0)
                            {
                                if ((studentsAnswers[i] == correctChoiceArray[j] && studentsAnswers[i + 1] == correctChoiceArray[j + 1]) ||
                                    (studentsAnswers[i + 1] == correctChoiceArray[j] && studentsAnswers[i] == correctChoiceArray[j + 1]))
                                {
                                    studentsCorrectAnswers += 2;
                                }
                            }
                        }
                    }

                    if (studentsCorrectAnswers == 0)
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Incorrect;
                    }
                    else
                    {
                        if (studentsReceivedPoints != subquestionPoints || studentsAnswers.Count != correctChoiceArray.Count)
                        {
                            isAnswerCorrect = StudentsAnswerCorrectness.PartiallyCorrect;
                        }
                        else
                        {
                            isAnswerCorrect = StudentsAnswerCorrectness.Correct;
                        }
                    }
                    break;
                case 5:
                    isAnswerCorrect = StudentsAnswerCorrectness.Unknown;
                    break;
                case 8:
                    if (correctAnswerArray[0] == studentsAnswers[0])
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Correct;
                    }
                    else
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Incorrect;
                    }
                    break;
            }

            switch (isAnswerCorrect)
            {
                case StudentsAnswerCorrectness.Correct:
                    //StudentsAnswerCorrectLabel.Text = "Správná odpověď.";
                    break;
                case StudentsAnswerCorrectness.PartiallyCorrect:
                    //StudentsAnswerCorrectLabel.Text = "Částečně správná odpověď.";
                    break;
                case StudentsAnswerCorrectness.Incorrect:
                    //StudentsAnswerCorrectLabel.Text = "Nesprávná odpověď.";
                    break;
                case StudentsAnswerCorrectness.Unknown:
                    //StudentsAnswerCorrectLabel.Text = "Otevřená odpověď, body budou přiděleny manuálně.";
                    break;
            }

            if ((studentsReceivedPoints < 0 && !negativePoints) || (studentsAnswers.Count > 0 && studentsAnswers[0] == ""))
            {
                studentsReceivedPoints = 0;
            }

            if (undecidedPointsInFile)
            {
                //StudentsAnswerPointstLabel.Text = "Počet bodů za odpověď: N/A";
            }
            else
            {
                //StudentsAnswerPointstLabel.Text = "Počet bodů za odpověď: " + studentsReceivedPoints + "/" + subquestionPoints;
            }
            studentsReceivedPointsArray.Add(Math.Round(studentsReceivedPoints, 2));
        }

        private void SaveResults(string testNameIdentifier, string resultNumberIdentifier, string resultPoints)
        {
            string file = Settings.GetResultResultsDataPath(testNameIdentifier, resultNumberIdentifier);
            File.WriteAllText(file, resultPoints);
        }

        public List<Result> Load(string testNameIdentifier)
        {
            List<Result> results = new List<Result>();

            foreach (var directory in Directory.GetDirectories(Settings.GetResultsPath()))
            {
                string resultNumberIdentifier = Path.GetFileName(directory);

                Result result = Load(testNameIdentifier, resultNumberIdentifier);
                results.Add(result);
            }

            return results;
        }

        public Result Load(string testNameIdentifier, string resultNumberIdentifier)
        {
            Result result = new Result();
            Test test = new TestData().Load(testNameIdentifier);

            byte lastRead = 0;

            // Load XML
            XmlReader xmlReader = XmlReader.Create(Settings.GetResultFilePath(testNameIdentifier, resultNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "testResult")
                {
                    if (xmlReader.NodeType != XmlNodeType.EndElement)
                    {
                        result.Datestamp = xmlReader.GetAttribute("datestamp");
                        lastRead = 1;
                    }
                }
            }

            // Results
            string resultsFilePath = Settings.GetResultResultsDataPath(testNameIdentifier, resultNumberIdentifier);
            bool resultsFileExists = false;
            
            foreach (var directory in Directory.GetDirectories(Settings.GetResultPath(testNameIdentifier)))
            {
                if (directory == resultsFilePath)
                {
                    resultsFileExists = true;
                    break;
                }
            }

            // #TODO
            if (!resultsFileExists)
            {
                string resultPointsToText = "";
                for (int i = 0; i < test.Items.Count; i++)
                {
                    string itemNameIdentifier = test.Items[i].Identifier;
                    string itemNumberIdentifier = test.Items[i].Href.Split("/")[3];
                    ItemForm itemForm = new ItemForm(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier, testNumberIdentifier, false, deliveryExecutionIdentifier, studentIdentifier, false, isTeacherReviewingDeliveryResult, negativePoints);
                    List<double> itemPoints = itemForm.GetResultsFilePoints();
                    // ItemForm()
                    // + LoadItemInfo(), LoadDeliveryExecutionInfo() nebo LoadDeliveryExecutionInfoToEdit()
                    // GetResultsFilePoints()
                    // + studentsReceivedPointsArray
                    //   + LoadDeliveryExecutionInfo()----------
                    List<double> itemPoints = LoadDeliveryExecutionInfo().studentsReceivedPointsArray;
                    resultPointsToText += itemNameIdentifier;
                    for (int j = 0; j < itemPoints.Count; j++)
                    {
                        resultPointsToText += ";" + Math.Round(itemPoints[j], 2);
                    }
                    resultPointsToText += "\n";
                }
                SaveResults(testNameIdentifier, resultNumberIdentifier, resultPointsToText);
            }

            string[] resultsFileLines = File.ReadAllLines(resultsFilePath);
            double studentsPoints = 0;
            for (int i = 0; i < resultsFileLines.Length; i++)
            {
                double studentsItemPoints = 0;
                string[] splitResultsFileLineBySemicolon = resultsFileLines[i].Split(";");
                for (int j = 1; j < splitResultsFileLineBySemicolon.Length; j++)
                {
                    studentsPoints += double.Parse(splitResultsFileLineBySemicolon[j]);
                    studentsItemPoints += double.Parse(splitResultsFileLineBySemicolon[j]);
                }
                ItemsGridView.Rows[i].Cells[4].Value = studentsItemPoints + "/" + test.Items[i].Points;
            }

            return result;
        }*/
    }
}
