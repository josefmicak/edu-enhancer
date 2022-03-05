using System.Xml;
using VDS.RDF;
using VDS.RDF.Parsing;

namespace ViewLayer.Controllers
{
    public class TestController
    {
        public string GetTestNumberIdentifier(string testNameIdentifier)
        {
            string testNumberIdentifier = "";
            foreach (var directory in Directory.GetDirectories("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\tests"))
            {
                testNumberIdentifier = ExtractFileName(directory);
            }
            return testNumberIdentifier;
        }

        public string ExtractFileExtension(string file)
        {
            string[] fileSplitByDot = file.Split(@".");
            return fileSplitByDot[fileSplitByDot.Length - 1];
        }

        public List<(string, string, string, string)> LoadTests(string studentIdentifier)
        {
            List<(string, string, string, string)> studentTestList = new List<(string, string, string, string)>();

            foreach (var directory in Directory.GetDirectories("C:\\xampp\\exported\\results"))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    string extension = ExtractFileExtension(file);
                    if (extension == "xml")
                    {
                        bool addTest = false;
                        string timeStamp = "";
                        string testStudentIdentifier = "";

                        XmlReader xmlReader = XmlReader.Create(file);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "context")
                            {
                                testStudentIdentifier = xmlReader.GetAttribute("sourcedId");
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
        }

        public string ExtractFileName(string file)
        {
            string[] fileSplitBySlash = file.Split(@"\");
            return fileSplitBySlash[fileSplitBySlash.Length - 1];
        }

        public string ExtractFileNameWithoutExtension(string file)
        {
            string[] fileSplitBySlash = file.Split(@"\");
            string[] fileSplitByDot = fileSplitBySlash[fileSplitBySlash.Length - 1].Split(@".");
            return fileSplitByDot[fileSplitByDot.Length - 2];
        }

        public (string, string, string, int) LoadTestParameters(string testNameIdentifier, string testNumberIdentifier)
        {
            string title = "";
            int amountOfItems = 0;

            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\tests\\" + testNumberIdentifier + "\\test.xml");
            while (xmlReader.Read())
            {
                if ((xmlReader.NodeType == XmlNodeType.Element) && (xmlReader.Name == "assessmentTest"))
                {
                    if (xmlReader.HasAttributes)
                    {
                        title = xmlReader.GetAttribute("title");
                    }
                }

                if (xmlReader.Name == "assessmentItemRef" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    amountOfItems++;
                }
            }
            return (testNameIdentifier, testNumberIdentifier, title, amountOfItems);
        }

        public List<(string, string, string, string)> LoadQuestions(string testNameIdentifier, string testNumberIdentifier)
        {
            List<(string, string, string, string)> questionList = new List<(string, string, string, string)>();
            string testPart = "";
            string testSection = "";
            string itemNameIdentifier = "";
            string itemNumberIdentifier = "";

            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\tests\\" + testNumberIdentifier + "\\test.xml");
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "testPart")
                {
                    testPart = xmlReader.GetAttribute("identifier");
                }

                if (xmlReader.Name == "assessmentSection")
                {
                    testSection = xmlReader.GetAttribute("identifier");
                }

                if (xmlReader.Name == "assessmentItemRef" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    itemNameIdentifier = xmlReader.GetAttribute("identifier");
                    string itemNumberIdentifierToSplit = xmlReader.GetAttribute("href");
                    string[] itemNumberIdentifierSplitBySlash = itemNumberIdentifierToSplit.Split(@"/");
                    itemNumberIdentifier = itemNumberIdentifierSplitBySlash[3];
                    questionList.Add((testPart, testSection, itemNameIdentifier, itemNumberIdentifier));
                }
            }

            return questionList;
        }

        public List<(string, string, string, string, int, bool)> LoadItemInfo(string testNameIdentifier, string testNumberIdentifier)//Jiné než v původní appce, načte seznam parametrů všech itemů
        {
            string title = "";
            string label = "";
            string itemNumberIdenfifier = "";
            string itemNameIdenfifier = "";
            int i = 0;
            List<(string, string, string, string, int, bool)> itemParametersTemp = new List<(string, string, string, string, int, bool)>();
            List<(string, string, string, string)> questionList = LoadQuestions(testNameIdentifier, testNumberIdentifier);
            bool testPointsDetermined = false;
            int testPoints = 0;

            foreach (var directory in Directory.GetDirectories("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\"))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    string[] fileSplitBySlash = file.Split(@"\");
                    if (fileSplitBySlash[fileSplitBySlash.Length - 1] != "qti.xml")
                    {
                        continue;
                    }
                    else
                    {
                        XmlReader xmlReader = XmlReader.Create(file);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                            {
                                if (xmlReader.Name == "assessmentItem")
                                {
                                    for (int j = 0; j < questionList.Count; j++)
                                    {
                                        if (questionList[j].Item4 == xmlReader.GetAttribute("identifier"))
                                        {
                                            itemNumberIdenfifier = questionList[j].Item4;
                                            itemNameIdenfifier = questionList[j].Item3;
                                            title = xmlReader.GetAttribute("title");
                                            label = xmlReader.GetAttribute("label");
                                            (int questionPoints, bool questionPointsDetermined) = GetQuestionPoints(testNameIdentifier, itemNumberIdenfifier);

                                            itemParametersTemp.Add((itemNumberIdenfifier, itemNameIdenfifier, title, label, questionPoints, questionPointsDetermined));
                                            i++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //pole nyní může být ve špatném pořadí kvůli jinému pořadí v rámci qti.xml celého testu a složek ve složce items - oprava
            List<(string, string, string, string, int, bool)> itemParameters = new List<(string, string, string, string, int, bool)>();
            for (int k = 0; k < questionList.Count; k++)
            {
                for (int l = 0; l < itemParametersTemp.Count; l++)
                {
                    if (questionList[k].Item4 == itemParametersTemp[l].Item1)
                    {
                        itemParameters.Add(itemParametersTemp[l]);
                    }
                }
            }

            return itemParameters;
        }

        public (int, bool) GetQuestionPoints(string testNameIdentifier, string itemNumberIdenfifier)
        {
            bool questionPointsDetermined = false;
            int questionPoints = 0;
            string itemParentPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdenfifier;

            foreach (var file in Directory.GetFiles(itemParentPath))
            {
                if (Path.GetFileName(file) == "Points.txt")
                {
                    questionPointsDetermined = true;
                }
            }

            if (questionPointsDetermined)
            {
                string[] importedFileLines = File.ReadAllLines(itemParentPath + "\\Points.txt");
                for (int j = 0; j < importedFileLines.Length; j++)
                {
                    string[] splitImportedFileLineBySemicolon = importedFileLines[j].Split(";");

                    if (splitImportedFileLineBySemicolon[1] == "N/A")
                    {
                        questionPointsDetermined = false;
                    }
                    else
                    {
                        questionPoints += int.Parse(splitImportedFileLineBySemicolon[1]);
                    }
                }
            }

            return (questionPoints, questionPointsDetermined);
        }

        public (int, bool) GetTestPoints(List<(string, string, string, string, int, bool)> itemParameters)
        {
            int testPoints = 0;
            bool testPointsDetermined = true;

            for (int i = 0; i < itemParameters.Count; i++)
            {
                testPoints += itemParameters[i].Item5;
                if (!itemParameters[i].Item6)
                {
                    testPointsDetermined = false;
                }
            }

            return (testPoints, testPointsDetermined);
        }

        public (List<(double, bool)>, int) GetQuestionResultPoints(List<(string, string, string, string, int, bool)> itemParameters, string testNameIdentifier, string testNumberIdentifier, string deliveryExecutionIdentifier)
        {
            ItemController itemController = new ItemController();
            List<(double, bool)> studentsAnswerPoints = new List<(double, bool)>();
            string resultsFilePath = "C:\\xampp\\exported\\results\\" + testNameIdentifier + "\\delivery_execution_" + deliveryExecutionIdentifier + "Results.txt";
            bool resultsFileExists = false;
            int errorMessageNumber = 0;

            foreach (var file in Directory.GetFiles("C:\\xampp\\exported\\results\\" + testNameIdentifier))
            {
                if (file == resultsFilePath)
                {
                    resultsFileExists = true;
                }
            }

            if (!resultsFileExists)
            {
                errorMessageNumber = itemController.CreateNewResultPointsFile(testNameIdentifier, testNumberIdentifier, deliveryExecutionIdentifier, itemParameters);
            }
            for (int i = 0; i < itemParameters.Count; i++)
            {
                string itemNameIdentifier = itemParameters[i].Item2;
                double totalReceivedPoints = 0;
                string[] resultsFileLines = File.ReadAllLines("C:\\xampp\\exported\\results\\" + testNameIdentifier + "\\delivery_execution_" + deliveryExecutionIdentifier + "Results.txt");
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

        }

        public double GetTotalStudentsPoints(List<(double, bool)> studentsPoints)
        {
            double totalStudentsPoints = 0;
            for (int i = 0; i < studentsPoints.Count; i++)
            {
                totalStudentsPoints += studentsPoints[i].Item1;
            }
            return totalStudentsPoints;
        }

        public (string, string, string, string, string) LoadResultParameters(string testNameIdentifier, string deliveryExecutionIdentifier, string studentIdentifier)
        {
            string resultTimestamp = "";
            string login = "", name = "", surname = "";

            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\results\\" + testNameIdentifier + "\\delivery_execution_" + deliveryExecutionIdentifier + ".xml");
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "testResult" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    resultTimestamp = xmlReader.GetAttribute("datestamp");
                }
            }

            foreach (var file in Directory.GetFiles("C:\\xampp\\exported\\testtakers"))
            {
                string extension = Path.GetExtension(file);
                if (extension == ".rdf")
                {
                    IGraph g = new Graph();
                    FileLoader.Load(g, file);
                    IEnumerable<INode> nodes = g.AllNodes;
                    int nodeLine = 1;
                    foreach (INode node in nodes)
                    {
                        if (nodeLine == 1)
                        {
                            string[] splitByHashtag = node.ToString().Split("#");
                            if (splitByHashtag[1] != studentIdentifier)
                            {
                                break;
                            }
                        }
                        if (nodeLine == 3)
                        {
                            login = node.ToString();
                        }
                        else if (nodeLine == 9)
                        {
                            name = node.ToString();
                        }
                        else if (nodeLine == 11)
                        {
                            surname = node.ToString();
                        }
                        nodeLine++;
                    }
                }
            }

            return (deliveryExecutionIdentifier, resultTimestamp, name + " " + surname, login, studentIdentifier);
        }

        public bool NegativePointsManagement(string testNameIdentifier, string testNumberIdentifier)
        {
            bool negativePointsInTest = false;
            bool fileExists = false;
            string testPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\tests\\" + testNumberIdentifier;

            foreach (var file in Directory.GetFiles(testPath))
            {
                if (Path.GetFileName(file) == "NegativePoints.txt")
                {
                    fileExists = true;
                    string[] negativePointsFileLines = File.ReadAllLines(file);
                    for (int i = 0; i < negativePointsFileLines.Length; i++)
                    {
                        if (negativePointsFileLines[0] == "1")
                        {
                            negativePointsInTest = true;
                        }
                    }
                }
            }

            if (!fileExists)
            {
                File.WriteAllText(testPath + "\\NegativePoints.txt", "0");
            }

            return negativePointsInTest;
        }
    }
}
