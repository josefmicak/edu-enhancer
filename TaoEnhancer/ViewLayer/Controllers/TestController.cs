using Microsoft.AspNetCore.Mvc;
using DomainModel;
using Common;
using System.Xml;

namespace ViewLayer.Controllers
{
    public class TestController : Controller
    {
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
                string title = "";
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
                                title = xmlReader.GetAttribute("title");
                                TestTemplate testTemplate = new TestTemplate();
                                testTemplate.testNameIdentifier = testNameIdentifier;
                                testTemplate.testNumberIdentifier = testNumberIdentifier;
                                testTemplate.title = title;
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
    }
}
