using Common;
using Common.Class;
using System.Xml;

namespace DataLayer
{
    public class TestData
    {
        StudentData studentData = new StudentData();

        /*----- Auxiliary functions -----*/
        private string ExtractFileName(string file)
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

        private string ExtractFileExtension(string file)
        {
            string[] fileSplitByDot = file.Split(@".");
            return fileSplitByDot[fileSplitByDot.Length - 1];
        }

        public string GetTestNumberIdentifier(string testNameIdentifier)
        {
            foreach (var directory in Directory.GetDirectories(Settings.GetTestTestsPath(testNameIdentifier)))
            {
                string testNumberIdentifier = Path.GetFileName(directory);

                return testNumberIdentifier;
            }
            throw new DirectoryNotFoundException();
        }

        /*----- File functions -----*/
        private bool HasNegativePoints(string testNameIdentifier)
        {
            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);
            if (File.Exists(Settings.GetTestTestNegativePointsDataPath(testNameIdentifier, testNumberIdentifier)))
            {
                string[] negativePointsFileLines = LoadNegativePoints(testNameIdentifier);
                if (negativePointsFileLines[0] == "1")
                {
                    return true;
                }
                return false;
            }
            return false;
        }

        private string[] LoadNegativePoints(string testNameIdentifier)
        {
            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);
            string file = Settings.GetTestTestNegativePointsDataPath(testNameIdentifier, testNumberIdentifier);
            return File.ReadAllLines(file);
        }

        private void SaveNegativePoints(string testNameIdentifier, string negativePoints)
        {
            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);
            string file = Settings.GetTestTestNegativePointsDataPath(testNameIdentifier, testNumberIdentifier);
            File.WriteAllText(file, negativePoints);
        }

        private bool UpdateNegativePoints(string testNameIdentifier)
        {
            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);
            if (File.Exists(Settings.GetTestTestNegativePointsDataPath(testNameIdentifier, testNumberIdentifier)))
            {
                string[] negativePointsFileLines = LoadNegativePoints(testNameIdentifier);
                if (negativePointsFileLines[0] == "1")
                {
                    return true;
                }
                return false;
            }
            SaveNegativePoints(testNameIdentifier, "0");
            return false;
        }

        // #DELETE
        /*private (string, string, string, int) LoadTestParameters(string testNameIdentifier)
        {
            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);
            string title = "";
            int amountOfItems = 0;

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestTestFilePath(testNameIdentifier, testNumberIdentifier));
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
        }*/

        // #DELETE
        /*private List<(string, string, string, string)> LoadTestQuestions(string testNameIdentifier)
        {
            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);
            List<(string, string, string, string)> questionList = new List<(string, string, string, string)>();
            string testPart = "";
            string testSection = "";

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
                    string itemNameIdentifier = xmlReader.GetAttribute("identifier");
                    string itemNumberIdentifierToSplit = xmlReader.GetAttribute("href");
                    string[] itemNumberIdentifierSplitBySlash = itemNumberIdentifierToSplit.Split(@"/");
                    string itemNumberIdentifier = itemNumberIdentifierSplitBySlash[3];
                    questionList.Add((testPart, testSection, itemNameIdentifier, itemNumberIdentifier));
                }
            }

            return questionList;
        }*/

        public (int points, bool pointsDetermined) GetTestPoints(List<Item> items)
        {
            int points = 0;
            bool pointsDetermined = true;

            foreach (Item item in items)
            {
                points += item.Points;
                if (!item.PointsDetermined)
                {
                    pointsDetermined = false;
                }
            }

            return (points, pointsDetermined);
        }

        /*----- Class functions -----*/
        public List<Test> Load()
        {
            List<Test> tests = new List<Test>();

            foreach (var directory in Directory.GetDirectories(Settings.GetTestsPath()))
            {
                string testNameIdentifier = Path.GetFileName(directory);

                Test test = Load(testNameIdentifier);
                tests.Add(test);
            }

            return tests;
        }

        public Test Load(string testNameIdentifier)
        {
            if(Directory.Exists(Settings.GetPath()))
            {
                if (Directory.Exists(Settings.GetTestsPath()))
                {
                    if (Directory.Exists(Settings.GetTestPath(testNameIdentifier)))
                    {
                        if (Directory.Exists(Settings.GetTestTestsPath(testNameIdentifier)))
                        {
                            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);
                            if (Directory.Exists(Settings.GetTestTestPath(testNameIdentifier, testNumberIdentifier)))
                            {
                                if (File.Exists(Settings.GetTestTestFilePath(testNameIdentifier, testNumberIdentifier)))
                                {
                                    Test test = null;
                                    TestPart part = null;
                                    TestSection section = null;
                                    TestItem item = null;

                                    byte lastRead = 0;

                                    XmlReader xmlReader = XmlReader.Create(Settings.GetTestTestFilePath(testNameIdentifier, testNumberIdentifier));
                                    while (xmlReader.Read())
                                    {
                                        if (xmlReader.Name == "assessmentTest")
                                        {
                                            if (xmlReader.NodeType != XmlNodeType.EndElement)
                                            {
                                                test = new Test(
                                                    xmlReader.GetAttribute("identifier"),
                                                    xmlReader.GetAttribute("title")//,
                                                                                   //xmlReader.GetAttribute("toolName"),
                                                    /*xmlReader.GetAttribute("toolVersion")*/);
                                                lastRead = 1;
                                            }
                                        }

                                        if (xmlReader.Name == "testPart")
                                        {
                                            if (xmlReader.NodeType != XmlNodeType.EndElement)
                                            {
                                                part = new TestPart(
                                                    xmlReader.GetAttribute("identifier"),
                                                    xmlReader.GetAttribute("navigationMode"),
                                                    xmlReader.GetAttribute("submissionMode"));
                                                lastRead = 2;
                                            }
                                            else if (test != null)
                                            {
                                                test.Parts.Add(part);
                                            }
                                        }

                                        if (xmlReader.Name == "assessmentSection")
                                        {
                                            if (xmlReader.NodeType != XmlNodeType.EndElement)
                                            {
                                                section = new TestSection(
                                                    xmlReader.GetAttribute("identifier"),
                                                    bool.Parse(xmlReader.GetAttribute("required")),
                                                    bool.Parse(xmlReader.GetAttribute("fixed")),
                                                    xmlReader.GetAttribute("title"),
                                                    bool.Parse(xmlReader.GetAttribute("visible")),
                                                    bool.Parse(xmlReader.GetAttribute("keepTogether")));
                                                lastRead = 3;
                                            }
                                            else if (part != null)
                                            {
                                                part.Sections.Add(section);
                                            }
                                        }

                                        if (xmlReader.Name == "assessmentItemRef")
                                        {
                                            if (xmlReader.NodeType != XmlNodeType.EndElement)
                                            {
                                                item = new TestItem(
                                                    xmlReader.GetAttribute("identifier"),
                                                    bool.Parse(xmlReader.GetAttribute("required")),
                                                    bool.Parse(xmlReader.GetAttribute("fixed")),
                                                    xmlReader.GetAttribute("href"));
                                                lastRead = 4;
                                            }
                                            else if (section != null)
                                            {
                                                section.Items.Add(item);
                                            }
                                        }

                                        if (xmlReader.Name == "itemSessionControl")
                                        {
                                            TestSessionControl sessionControl = new TestSessionControl(
                                                int.Parse(xmlReader.GetAttribute("maxAttempts")),
                                                bool.Parse(xmlReader.GetAttribute("showFeedback")),
                                                bool.Parse(xmlReader.GetAttribute("allowReview")),
                                                bool.Parse(xmlReader.GetAttribute("showSolution")),
                                                bool.Parse(xmlReader.GetAttribute("allowComment")),
                                                bool.Parse(xmlReader.GetAttribute("allowSkipping")),
                                                bool.Parse(xmlReader.GetAttribute("validateResponses")));

                                            switch (lastRead)
                                            {
                                                case 1:
                                                    test.SessionControl = sessionControl;
                                                    break;
                                                case 2:
                                                    part.SessionControl = sessionControl;
                                                    break;
                                                case 3:
                                                    section.SessionControl = sessionControl;
                                                    break;
                                                case 4:
                                                    item.SessionControl = sessionControl;
                                                    break;
                                            }
                                            lastRead = 5;
                                        }

                                        if (xmlReader.Name == "timeLimits")
                                        {
                                            TestTimeLimits timeLimits = new TestTimeLimits(
                                                bool.Parse(xmlReader.GetAttribute("allowLateSubmission")));

                                            switch (lastRead)
                                            {
                                                case 1:
                                                    test.TimeLimits = timeLimits;
                                                    break;
                                                case 2:
                                                    part.TimeLimits = timeLimits;
                                                    break;
                                                case 3:
                                                    section.TimeLimits = timeLimits;
                                                    break;
                                                case 4:
                                                    item.TimeLimits = timeLimits;
                                                    break;
                                            }
                                            lastRead = 6;
                                        }
                                    }

                                    test.NegativePoints = UpdateNegativePoints(testNameIdentifier);

                                    return test;
                                }
                                else { throw Exceptions.TestTestFilePathNotFoundException; }
                            }
                            else { throw Exceptions.TestTestPathNotFoundException; }
                        }
                        else { throw Exceptions.TestTestsPathNotFoundException; }
                    }
                    else { throw Exceptions.TestPathNotFoundException; }
                }
                else { throw Exceptions.TestsPathNotFoundException; }
            }
            else { throw Exceptions.PathNotFoundException; }
        }

        //-------------------------------------------------- #OLD --------------------------------------------------

        /*private void SaveNegativePoints(string testNameIdentifier, string negativePoints)
        {
            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);

            string file = Settings.GetTestTestNegativePointsDataPath(testNameIdentifier, testNumberIdentifier);
            File.WriteAllText(file, negativePoints);
        }

        public string GetTestNumberIdentifier(string testNameIdentifier)
        {
            foreach (var directory in Directory.GetDirectories(Settings.GetTestTestsPath(testNameIdentifier)))
            {
                string testNumberIdentifier = Path.GetFileName(directory);

                return testNumberIdentifier;
            }
            throw new DirectoryNotFoundException();
        }

        public List<Test> Load()
        {
            List<Test> tests = new List<Test>();

            foreach (var directory in Directory.GetDirectories(Settings.GetTestsPath()))
            {
                string testNameIdentifier = Path.GetFileName(directory);

                Test test = Load(testNameIdentifier);
                tests.Add(test);
            }

            return tests;
        }

        public Test Load(string testNameIdentifier)
        {
            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);

            Test test = null;
            TestPart part = null;
            TestSection section = null;
            TestItem item = null;

            byte lastRead = 0;

            // Load XML
            XmlReader xmlReader = XmlReader.Create(Settings.GetTestTestFilePath(testNameIdentifier, testNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "assessmentTest")
                {
                    if (xmlReader.NodeType != XmlNodeType.EndElement)
                    {
                        test = new Test(
                            xmlReader.GetAttribute("identifier"),
                            xmlReader.GetAttribute("title"),
                            xmlReader.GetAttribute("toolName"),
                            xmlReader.GetAttribute("toolVersion"));
                        lastRead = 1;
                    }
                }

                if (xmlReader.Name == "testPart")
                {
                    if (xmlReader.NodeType != XmlNodeType.EndElement)
                    {
                        part = new TestPart(
                            xmlReader.GetAttribute("identifier"),
                            xmlReader.GetAttribute("navigationMode"),
                            xmlReader.GetAttribute("submissionMode"));
                        lastRead = 2;
                    }
                    else if (test != null)
                    {
                        test.Parts.Add(part);
                    }
                }

                if (xmlReader.Name == "assessmentSection")
                {
                    if (xmlReader.NodeType != XmlNodeType.EndElement)
                    {
                        section = new TestSection(
                            xmlReader.GetAttribute("identifier"),
                            bool.Parse(xmlReader.GetAttribute("required")),
                            bool.Parse(xmlReader.GetAttribute("fixed")),
                            xmlReader.GetAttribute("title"),
                            bool.Parse(xmlReader.GetAttribute("visible")),
                            bool.Parse(xmlReader.GetAttribute("keepTogether")));
                        lastRead = 3;
                    }
                    else if (part != null)
                    {
                        part.Sections.Add(section);
                    }
                }

                if (xmlReader.Name == "assessmentItemRef")
                {
                    if (xmlReader.NodeType != XmlNodeType.EndElement)
                    {
                        item = new TestItem(
                            xmlReader.GetAttribute("identifier"),
                            bool.Parse(xmlReader.GetAttribute("required")),
                            bool.Parse(xmlReader.GetAttribute("fixed")),
                            xmlReader.GetAttribute("href"));
                        lastRead = 4;
                    }
                    else if (section != null)
                    {
                        section.Items.Add(item);
                    }
                }

                if (xmlReader.Name == "itemSessionControl")
                {
                    TestSessionControl sessionControl = new TestSessionControl(
                        int.Parse(xmlReader.GetAttribute("maxAttempts")),
                        bool.Parse(xmlReader.GetAttribute("showFeedback")),
                        bool.Parse(xmlReader.GetAttribute("allowReview")),
                        bool.Parse(xmlReader.GetAttribute("showSolution")),
                        bool.Parse(xmlReader.GetAttribute("allowComment")),
                        bool.Parse(xmlReader.GetAttribute("allowSkipping")),
                        bool.Parse(xmlReader.GetAttribute("validateResponses")));

                    switch (lastRead)
                    {
                        case 1:
                            test.SessionControl = sessionControl;
                            break;
                        case 2:
                            part.SessionControl = sessionControl;
                            break;
                        case 3:
                            section.SessionControl = sessionControl;
                            break;
                        case 4:
                            item.SessionControl = sessionControl;
                            break;
                    }
                    lastRead = 5;
                }

                if (xmlReader.Name == "timeLimits")
                {
                    TestTimeLimits timeLimits = new TestTimeLimits(
                        bool.Parse(xmlReader.GetAttribute("allowLateSubmission")));

                    switch (lastRead)
                    {
                        case 1:
                            test.TimeLimits = timeLimits;
                            break;
                        case 2:
                            part.TimeLimits = timeLimits;
                            break;
                        case 3:
                            section.TimeLimits = timeLimits;
                            break;
                        case 4:
                            item.TimeLimits = timeLimits;
                            break;
                    }
                    lastRead = 6;
                }
            }

            // Points
            for (int i = 0; i < test.Parts.Count; i++)
            {
                for (int j = 0; j < test.Parts[i].Sections.Count; j++)
                {
                    for (int k = 0; k < test.Parts[i].Sections[j].Items.Count; k++)
                    {
                        foreach (var file in Directory.GetFiles(Settings.GetTestItemPath(testNameIdentifier, test.Items[k].Href.Split("/")[3])))
                        {
                            if (Path.GetFileName(file) == "Points.txt")
                            {
                                test.Parts[i].Sections[j].Items[k].PointsDetermined = true;

                                string[] importedFileLines = File.ReadAllLines(file);
                                for (int l = 0; l < importedFileLines.Length; l++)
                                {
                                    string[] splitImportedFileLineBySemicolon = importedFileLines[l].Split(";");

                                    if (splitImportedFileLineBySemicolon[1] == "N/A")
                                    {
                                        test.Parts[i].Sections[j].Items[k].PointsDetermined = false;
                                    }
                                    else
                                    {
                                        test.Parts[i].Sections[j].Items[k].Points += int.Parse(splitImportedFileLineBySemicolon[1]);
                                    }
                                }

                                break;
                            }
                        }

                        if (!test.Parts[i].Sections[j].Items[k].PointsDetermined)
                        {
                            test.PointsDetermined = false;
                        }
                    }
                }
            }

            // Negative points
            bool fileExists = false;
            foreach (var file in Directory.GetFiles(Settings.GetTestTestPath(testNameIdentifier, testNumberIdentifier)))
            {
                if (Path.GetFileName(file) == "NegativePoints.txt")
                {
                    fileExists = true;
                    string[] negativePointsFileLines = File.ReadAllLines(file);
                    if (negativePointsFileLines[0] == "1")
                    {
                        test.NegativePoints = true;
                    }

                    break;
                }
            }
            if (!fileExists)
            {
                SaveNegativePoints(testNameIdentifier, "0");
            }

            return test;
        }*/
    }
}
