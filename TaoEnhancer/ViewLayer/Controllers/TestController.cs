using Microsoft.AspNetCore.Mvc;
using DomainModel;
using Common;
using System.Xml;
using System.Diagnostics;
using DataLayer;
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
        public List<TestTemplate> LoadTestTemplates(string login)
        {
            List<TestTemplate> testTemplates = new List<TestTemplate>();
            string subDirectory = "";

            if (Directory.Exists(Config.GetTestTemplatesPath()))
            {
                foreach (var directory in Directory.GetDirectories(Config.GetTestTemplatesPath()))
                {
                    string[] splitDirectoryBySlash = directory.Split(Config.GetPathSeparator());
                    string testNameIdentifier = splitDirectoryBySlash[splitDirectoryBySlash.Length - 1].ToString();
                    string testNumberIdentifier = "";

                    try
                    {
                        foreach (var directory_ in Directory.GetDirectories(directory + Config.GetPathSeparator() + "tests"))
                        {
                            string[] splitDirectory_BySlash = directory_.Split(Config.GetPathSeparator());
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
                        XmlReader xmlReader = XmlReader.Create(subDirectory + Config.GetPathSeparator() + "test.xml");
                        while (xmlReader.Read())
                        {
                            if ((xmlReader.NodeType == XmlNodeType.Element) && (xmlReader.Name == "assessmentTest"))
                            {
                                if (xmlReader.HasAttributes)
                                {
                                    TestTemplate testTemplate = new TestTemplate();
                                    testTemplate.TestNameIdentifier = testNameIdentifier;
                                    testTemplate.TestNumberIdentifier = testNumberIdentifier;
                                    if(xmlReader.GetAttribute("title") != null)
                                    {
                                        testTemplate.Title = xmlReader.GetAttribute("title")!;
                                    }
                                    testTemplate.OwnerLogin = login;
                                    if(_context.Users.First(u => u.Login == login) != null)
                                    {
                                        testTemplate.Owner = _context.Users.First(u => u.Login == login);
                                    }
                                    else
                                    {
                                        throw Exceptions.SpecificUserNotFoundException(login);
                                    }
                                    testTemplate.QuestionTemplateList = questionController.LoadQuestionTemplates(testTemplate, login);
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
            else
            {
                throw Exceptions.TestTemplatesPathNotFoundException;
            }
        }

        /// <summary>
        /// Returns the list of test results
        /// </summary>
        /// <returns>the list of test results</returns>
        public List<TestResult> LoadTestResults(string login)
        {
            List<TestResult> testResults = new List<TestResult>();

            if (Directory.Exists(Config.GetResultsPath()))
            {
                foreach (var directory in Directory.GetDirectories(Config.GetResultsPath()))
                {
                    foreach (var file in Directory.GetFiles(directory))
                    {
                        if (Path.GetExtension(file) == ".xml")
                        {
                            string timeStampString = "";
                            string testStudentIdentifier = "";

                            XmlReader xmlReader = XmlReader.Create(file);
                            while (xmlReader.Read())
                            {
                                if (xmlReader.Name == "context")
                                {
                                    if(xmlReader.GetAttribute("sourcedId") != null)
                                    {
                                        testStudentIdentifier = xmlReader.GetAttribute("sourcedId")!;
                                    }
                                }

                                if (xmlReader.Name == "testResult")
                                {
                                    if(xmlReader.GetAttribute("datestamp") != null)
                                    {
                                        timeStampString = xmlReader.GetAttribute("datestamp")!;
                                    }
                                }
                            }
                            string[] attemptIdentifierSplitByUnderscore = Path.GetFileNameWithoutExtension(file).Split("_");
                            TestResult testResult = new TestResult();
                            testResult.TestResultIdentifier = attemptIdentifierSplitByUnderscore[2];

                            if(Path.GetFileName(Path.GetDirectoryName(file)) != null)
                            {
                                testResult.TestNameIdentifier = Path.GetFileName(Path.GetDirectoryName(file))!;
                            }
                            else
                            {
                                throw Exceptions.TestTemplateNotFoundException(testResult.TestResultIdentifier);
                            }

                            if (_context.TestTemplates.Count() == 0)
                            {
                                throw Exceptions.TestTemplatesNotImportedException;
                            }
                            if (_context.TestTemplates.Include(t => t.QuestionTemplateList)
                                .FirstOrDefault(t => t.TestNameIdentifier == testResult.TestNameIdentifier && t.OwnerLogin == login) != null)
                            {
                                testResult.TestTemplate = _context.TestTemplates.Include(t => t.QuestionTemplateList)
                                    .First(t => t.TestNameIdentifier == testResult.TestNameIdentifier && t.OwnerLogin == login);
                            }
                            else
                            {
                                throw Exceptions.TestTemplateNotFoundException(testResult.TestResultIdentifier);
                            }

                            testResult.TestNumberIdentifier = testResult.TestTemplate.TestNumberIdentifier;

                            if(_context.Students.Count() == 0)
                            {
                                throw Exceptions.StudentsNotImportedException;
                            }
                            if(_context.Students.FirstOrDefault(s => s.StudentIdentifier == testStudentIdentifier) != null)
                            {
                                testResult.Student = _context.Students.First(s => s.StudentIdentifier == testStudentIdentifier);
                                testResult.StudentLogin = testResult.Student.Login;
                            }
                            else
                            {
                                throw Exceptions.StudentNotFoundException(testStudentIdentifier);
                            }

                            testResult.OwnerLogin = login;

                            DateTime timeStamp = DateTime.ParseExact(timeStampString, "yyyy-MM-ddTHH:mm:ss.fff",
                                    System.Globalization.CultureInfo.InvariantCulture);
                            testResult.TimeStamp = timeStamp;

                            testResult.QuestionResultList = questionController.LoadQuestionResults(testResult, login);

                            testResults.Add(testResult);
                        }
                    }
                }
                return testResults;
            }
            else
            {
                throw Exceptions.TestResultsPathNotFoundException;
            }

        }
    }
}
