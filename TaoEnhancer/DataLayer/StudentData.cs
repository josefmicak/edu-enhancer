using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataLayer
{
    public class StudentData
    {
        public List<TestTaker> LoadTestTakers()
        {
            List<TestTaker> testTakers = new List<TestTaker>();

            foreach (var file in Directory.GetFiles(Settings.Path + "\\testtakers"))
            {
                string testTakerNumberIdentifier = Path.GetFileName(file);

                TestTaker testTaker = LoadTestTaker(testTakerNumberIdentifier);
                testTakers.Add(testTaker);
            }

            return testTakers;
        }

        public TestTaker LoadTestTaker(string testTakerNumberIdentifier)
        {
            TestTaker testTaker = new TestTaker();

            // Load XML
            XmlReader xmlReader = XmlReader.Create(Settings.Path + "\\testtakers\\" + testTakerNumberIdentifier);
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    xmlReader.MoveToAttribute("rdf:about");
                    testTaker.Identifier = xmlReader.GetAttribute("rdf:about").Split("#")[1];
                }

                if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testTaker.Login = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:password" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testTaker.Password = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userDefLg" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testTaker.UserDefaultLanguage = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userFirstName" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testTaker.FirstName = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userLastName" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testTaker.LastName = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userMail" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testTaker.UserMail = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userRoles" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testTaker.UserRoles = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userUILg" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testTaker.UserUILanguage = xmlReader.ReadInnerXml();
                }
            }

            return testTaker;
        }
    }
}
