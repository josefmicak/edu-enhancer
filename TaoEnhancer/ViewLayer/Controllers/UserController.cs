using Microsoft.AspNetCore.Mvc;
using DomainModel;
using Common;
using System.Xml;

namespace ViewLayer.Controllers
{
    public class UserController : Controller
    {
        public List<User> LoadStudents()
        {
            List<User> students = new List<User>();
            if (Directory.Exists(Settings.GetStudentsPath()))
            {
                foreach (var studentFile in Directory.GetFiles(Settings.GetStudentsPath()))
                {
                    if (new FileInfo(studentFile).Extension == ".rdf")
                    {
                        User student = new User();
                        XmlReader xmlReader = XmlReader.Create(studentFile);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.UserIdentifier = xmlReader.GetAttribute("rdf:about").Split("#")[1];
                            }

                            if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.Login = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userFirstName" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.FirstName = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userLastName" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.LastName = xmlReader.ReadInnerXml();
                            }

                            student.Role = 1;
                        }
                        students.Add(student);
                    }
                }
            }
            else { throw Exceptions.StudentsPathNotFoundException; }
            return students;
        }

        /// <summary>
        /// Returns the selected student
        /// </summary>
        /// <param name="studentIdentifier">Identifier of the selected student</param>
        /// <returns>the selected test student</returns>
        public User LoadStudent(string studentIdentifier)
        {
            User student = new User();

            if (Directory.Exists(Settings.GetStudentsPath()))
            {
                foreach (var studentFile in Directory.GetFiles(Settings.GetStudentsPath()))
                {
                    if (new FileInfo(studentFile).Extension == ".rdf")
                    {
                        XmlReader xmlReader = XmlReader.Create(studentFile);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                if (studentIdentifier == xmlReader.GetAttribute("rdf:about").Split("#")[1])
                                {
                                    student.UserIdentifier = xmlReader.GetAttribute("rdf:about").Split("#")[1];
                                }
                                else
                                {
                                    break;
                                }
                            }

                            if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.Login = xmlReader.ReadInnerXml();
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
                                student.Email = xmlReader.ReadInnerXml();
                            }
                        }
                    }
                }
            }
            else { throw Exceptions.StudentsPathNotFoundException; }
            return student;
        }
    }
}
