using Common;
using Common.Class;
using System.Xml;

namespace DataLayer
{
    public class StudentData
    {
        /*----- File functions -----*/
        private List<Student> Load()
        {
            List<Student> students = new List<Student>();

            foreach (var file in Directory.GetFiles(Settings.GetStudentsPath()))
            {
                string studentNumberIdentifier = Path.GetFileName(file);

                Student student = Load(studentNumberIdentifier);
                students.Add(student);
            }

            return students;
        }

        public Student Load(string studentIdentifier)
        {
            Student student = new Student();

            // Load XML
            XmlReader xmlReader = XmlReader.Create(Settings.GetStudentFilePath(studentIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    xmlReader.MoveToAttribute("rdf:about");
                    student.Identifier = xmlReader.GetAttribute("rdf:about").Split("#")[1];
                }

                if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.Login = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:password" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.Password = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userDefLg" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.UserDefaultLanguage = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userFirstName" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.FirstName = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userLastName" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.LastName = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userMail" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.UserMail = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userRoles" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.UserRoles = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userUILg" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.UserUILanguage = xmlReader.ReadInnerXml();
                }
            }

            return student;
        }

        //-------------------------------------------------- #OLD --------------------------------------------------

        /*public List<Student> Load()
        {
            List<Student> students = new List<Student>();

            foreach (var file in Directory.GetFiles(Settings.GetStudentsPath()))
            {
                string studentNumberIdentifier = Path.GetFileName(file);

                Student student = Load(studentNumberIdentifier);
                students.Add(student);
            }

            return students;
        }

        public Student Load(string studentNumberIdentifier)
        {
            Student student = new Student();

            // Load XML
            XmlReader xmlReader = XmlReader.Create(Settings.GetStudentFilePath(studentNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    xmlReader.MoveToAttribute("rdf:about");
                    student.Identifier = xmlReader.GetAttribute("rdf:about").Split("#")[1];
                }

                if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.Login = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:password" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.Password = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userDefLg" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.UserDefaultLanguage = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userFirstName" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.FirstName = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userLastName" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.LastName = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userMail" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.UserMail = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userRoles" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.UserRoles = xmlReader.ReadInnerXml();
                }

                if (xmlReader.Name == "ns0:userUILg" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    student.UserUILanguage = xmlReader.ReadInnerXml();
                }
            }

            return student;
        }*/
    }
}
