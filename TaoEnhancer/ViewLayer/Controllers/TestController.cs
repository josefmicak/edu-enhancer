using Common;
using System.Xml;
using VDS.RDF;
using VDS.RDF.Parsing;

namespace ViewLayer.Controllers
{
    public class TestController
    {
        public (string, string, string, int) LoadTestParameters(string testNameIdentifier, string testNumberIdentifier)
        {
            string title = "";
            int amountOfItems = 0;

            if (Directory.Exists(Settings.GetTestPath(testNameIdentifier)))
            {
                if (Directory.Exists(Settings.GetTestTestsPath(testNameIdentifier)))
                {
                    if (Directory.Exists(Settings.GetTestTestPath(testNameIdentifier, testNumberIdentifier)))
                    {
                        if (File.Exists(Settings.GetTestTestFilePath(testNameIdentifier, testNumberIdentifier)))
                        {
                            try
                            {
                                XmlReader xmlReader = XmlReader.Create(Settings.GetTestTestFilePath(testNameIdentifier, testNumberIdentifier));
                                while (xmlReader.Read())
                                {
                                    if (xmlReader.Name == "assessmentTest" && xmlReader.NodeType != XmlNodeType.EndElement)
                                    {
                                        if (xmlReader.GetAttribute("title") != null)
                                        {
                                            title = xmlReader.GetAttribute("title");
                                        }
                                        else
                                        {
                                            throw Exceptions.XmlAttributeNotFound;
                                        }
                                    }

                                    if (xmlReader.Name == "assessmentItemRef" && xmlReader.NodeType != XmlNodeType.EndElement)
                                    {
                                        amountOfItems++;
                                    }
                                }
                            }
                            catch (XmlException e) when (e.Message == "Root element is missing.")
                            {
                                throw Exceptions.XmlRootElementMissing;
                            }
                        }
                        else { throw Exceptions.TestTestFilePathNotFoundException; }
                    }
                    else { throw Exceptions.TestTestPathNotFoundException; }
                }
                else { throw Exceptions.TestTestsPathNotFoundException; }
            }
            else { throw Exceptions.TestPathNotFoundException; }

            return (testNameIdentifier, testNumberIdentifier, title, amountOfItems);
        }

        public List<(string, string, string, string)> LoadQuestions(string testNameIdentifier, string testNumberIdentifier)
        {
            List<(string, string, string, string)> questionList = new List<(string, string, string, string)>();
            string testPart = "";
            string testSection = "";
            string itemNameIdentifier = "";
            string itemNumberIdentifier = "";

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestTestFilePath(testNameIdentifier, testNumberIdentifier));
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

            foreach (var directory in Directory.GetDirectories(Settings.GetTestItemsPath(testNameIdentifier)))
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

            if (Directory.Exists(Settings.GetTestPath(testNameIdentifier)))
            {
                if (Directory.Exists(Settings.GetTestItemsPath(testNameIdentifier)))
                {
                    if (Directory.Exists(Settings.GetTestItemPath(testNameIdentifier, itemNumberIdenfifier)))
                    {
                        if (File.Exists(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdenfifier)))
                        {
                            string[] importedFileLines = File.ReadAllLines(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdenfifier));
                            foreach (string importedFileLine in importedFileLines)
                            {
                                if (importedFileLine.Contains(';'))
                                {
                                    string[] splitImportedFileLineBySemicolon = importedFileLine.Split(";");

                                    if (splitImportedFileLineBySemicolon[0].Length > 0)
                                    {
                                        try
                                        {
                                            questionPoints += int.Parse(splitImportedFileLineBySemicolon[1]);
                                            questionPointsDetermined = true;
                                        }
                                        catch (Exception e)
                                        {
                                            questionPointsDetermined = false;
                                            break;
                                        }
                                    }
                                    else
                                    {
                                        throw Exceptions.DataIdentifierNotFound;
                                    }
                                }
                                else
                                {
                                    questionPointsDetermined = false;
                                    break;
                                }
                            }
                        }
                    }
                    else { throw Exceptions.TestItemPathNotFoundException; }
                }
                else { throw Exceptions.TestItemsPathNotFoundException; }
            }
            else { throw Exceptions.TestPathNotFoundException; }

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
            bool resultsFileExists = false;
            int errorMessageNumber = 0;

            if(File.Exists(Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier)))
            {
                resultsFileExists = true;
            }

            if (!resultsFileExists)
            {
                errorMessageNumber = itemController.CreateNewResultPointsFile(testNameIdentifier, testNumberIdentifier, deliveryExecutionIdentifier, itemParameters);
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

            XmlReader xmlReader = XmlReader.Create(Settings.GetResultFilePath(testNameIdentifier, deliveryExecutionIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "testResult" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    resultTimestamp = xmlReader.GetAttribute("datestamp");
                }
            }

            foreach (var file in Directory.GetFiles(Settings.GetStudentsPath()))
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
            
            if(File.Exists(Settings.GetTestTestNegativePointsDataPath(testNameIdentifier, testNumberIdentifier)))
            {
                fileExists = true;
                string[] negativePointsFileLines = File.ReadAllLines(Settings.GetTestTestNegativePointsDataPath(testNameIdentifier, testNumberIdentifier));
                for (int i = 0; i < negativePointsFileLines.Length; i++)
                {
                    if (negativePointsFileLines[0] == "1")
                    {
                        negativePointsInTest = true;
                    }
                }
            }

            if (!fileExists)
            {
                File.WriteAllText(Settings.GetTestTestNegativePointsDataPath(testNameIdentifier, testNumberIdentifier), "0");
            }

            return negativePointsInTest;
        }
    }
}
