using Microsoft.AspNetCore.Mvc;
using DomainModel;
using Common;
using System.Xml;

namespace ViewLayer.Controllers
{
    public class TestController : Controller
    {
        private QuestionController questionController = new QuestionController();

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
                                //TODO: Mozna uncomment
                                //testTemplate.questionTemplateList = LoadQuestionTemplates(testNameIdentifier, testNumberIdentifier);
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
        /// Returns the selected test template
        /// </summary>
        /// <param name="selectedTestNameIdentifier">Name identifier of the selected test</param>
        /// <param name="selectedTestNumberIdentifier">Number identifier of the selected test</param>
        /// <returns>the selected test template</returns>
        public TestTemplate LoadTestTemplate(string selectedTestNameIdentifier, string selectedTestNumberIdentifier)
        {
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
                                if(selectedTestNameIdentifier == testNameIdentifier && selectedTestNumberIdentifier == testNumberIdentifier)
                                {
                                    TestTemplate testTemplate = new TestTemplate();
                                    testTemplate.TestNameIdentifier = testNameIdentifier;
                                    testTemplate.TestNumberIdentifier = testNumberIdentifier;
                                    testTemplate.Title = xmlReader.GetAttribute("title");
                                    testTemplate.QuestionTemplateList = questionController.LoadQuestionTemplates(testNameIdentifier, testNumberIdentifier);
                                    return testTemplate;
                                }
                                //TODO: Mozna uncomment
                                //testTemplate.questionTemplateList = LoadQuestionTemplates(testNameIdentifier, testNumberIdentifier);
                            }
                        }
                    }
                }
                catch
                {
                    continue;
                }
            }
            return null;
        }
    }
}
