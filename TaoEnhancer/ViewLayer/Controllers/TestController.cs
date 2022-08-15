using Microsoft.AspNetCore.Mvc;
using DomainModel;
using Common;
using System.Xml;
using System.Diagnostics;
using ViewLayer.Data;
using Microsoft.EntityFrameworkCore;

namespace ViewLayer.Controllers
{
    public class TestController : Controller
    {
        private readonly CourseContext _context;
        private QuestionController questionController;
        private StudentController studentController = new StudentController();

        public TestController(CourseContext context)
        {
            _context = context;
            questionController = new QuestionController(context);
        }

        /// <summary>
        /// Returns the list of test templates
        /// </summary>
        /// <returns>the list of test templates</returns>
        public List<TestTemplate> LoadTestTemplates()
        {
            List<TestTemplate> testTemplates = new List<TestTemplate>();
            string subDirectory = "";

            foreach (var directory in Directory.GetDirectories(Settings.GetTestsPath()))
            {
                string[] splitDirectoryBySlash = directory.Split(Settings.GetPathSeparator());
                string testNameIdentifier = splitDirectoryBySlash[splitDirectoryBySlash.Length - 1].ToString();
                string testNumberIdentifier = "";

                try
                {
                    foreach (var directory_ in Directory.GetDirectories(directory + Settings.GetPathSeparator() + "tests"))
                    {
                        string[] splitDirectory_BySlash = directory_.Split(Settings.GetPathSeparator());
                        testNumberIdentifier = splitDirectory_BySlash[splitDirectory_BySlash.Length - 1].ToString();
                        subDirectory = directory_;
                    }
                }
                catch
                {
                    continue;
                }

                try
                {
                    XmlReader xmlReader = XmlReader.Create(subDirectory + Settings.GetPathSeparator() + "test.xml");
                    while (xmlReader.Read())
                    {
                        if ((xmlReader.NodeType == XmlNodeType.Element) && (xmlReader.Name == "assessmentTest"))
                        {
                            if (xmlReader.HasAttributes)
                            {
                                TestTemplate testTemplate = new TestTemplate();
                                testTemplate.TestNameIdentifier = testNameIdentifier;
                                testTemplate.TestNumberIdentifier = testNumberIdentifier;
                                testTemplate.Title = xmlReader.GetAttribute("title");
                                testTemplate.QuestionTemplateList = questionController.LoadQuestionTemplates(testTemplate);
                                testTemplates.Add(testTemplate);
                            }
                        }
                    }
                }
                catch
                {
                    continue;
                }
            }

            return testTemplates;
        }

        /// <summary>
        /// Returns the list of test results
        /// </summary>
        /// <returns>the list of test results</returns>
        public List<TestResult> LoadTestResults()
        {
            List<TestResult> testResults = new List<TestResult>();

            foreach (var directory in Directory.GetDirectories(Settings.GetResultsPath()))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    if (Path.GetExtension(file) == ".xml")
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
                        string[] attemptIdentifierSplitByUnderscore = Path.GetFileNameWithoutExtension(file).Split("_");
                        TestResult testResult = new TestResult();
                        testResult.TestResultIdentifier = attemptIdentifierSplitByUnderscore[2];
                        testResult.TestNameIdentifier = Path.GetFileName(Path.GetDirectoryName(file));
                        testResult.TestTemplate = _context.TestTemplates.Include(t => t.QuestionTemplateList).FirstOrDefault(t => t.TestNameIdentifier == testResult.TestNameIdentifier);
                        _context.TestTemplates.Attach(testResult.TestTemplate);
                        testResult.Student = studentController.LoadStudent(testStudentIdentifier);
                        testResult.TimeStamp = timeStamp;
                        testResult.QuestionResultList = questionController.LoadQuestionResults(testResult, testResult.TestTemplate);
                        testResults.Add(testResult);
                    }
                }
            }
            return testResults;
        }
    }
}
