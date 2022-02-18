using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataLayer
{
    public class TestData
    {
        public string GetTestNumberIdentifier(string testNameIdentifier)
        {
            foreach (var directory in Directory.GetDirectories(Settings.Path + "\\tests\\" + testNameIdentifier + "\\tests"))
            {
                string testNumberIdentifier = Path.GetFileName(directory);

                return testNumberIdentifier;
            }
            throw new DirectoryNotFoundException();
        }

        public List<Test> LoadTests()
        {
            List<Test> tests = new List<Test>();

            foreach (var directory in Directory.GetDirectories(Settings.Path + "\\tests"))
            {
                string testNameIdentifier = Path.GetFileName(directory);

                Test test = LoadTest(testNameIdentifier);
                tests.Add(test);
            }

            return tests;
        }

        public Test LoadTest(string testNameIdentifier)
        {
            string testNumberIdentifier = GetTestNumberIdentifier(testNameIdentifier);

            Test test = null;
            TestPart part = null;
            TestSection section = null;
            TestItem item = null;

            byte lastRead = 0;

            // Load XML
            XmlReader xmlReader = XmlReader.Create(Settings.Path + "\\tests\\" + testNameIdentifier + "\\tests\\" + testNumberIdentifier + "\\test.xml");
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
                        foreach (var file in Directory.GetFiles(Settings.Path + "\\tests\\" + testNameIdentifier + "\\items\\" + test.Items[k].Href.Split("/")[3]))
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
            foreach (var file in Directory.GetFiles(Settings.Path + "\\tests\\" + testNameIdentifier + "\\tests\\" + testNumberIdentifier))
            {
                if (Path.GetFileName(file) == "NegativePoints.txt")
                {
                    string[] negativePointsFileLines = File.ReadAllLines(file);
                    if (negativePointsFileLines[0] == "1")
                    {
                        test.NegativePoints = true;
                    }
                }
            }

            return test;
        }
    }
}
