using Microsoft.AspNetCore.Mvc;
using DomainModel;
using Common;
using System.Xml;

namespace ViewLayer.Controllers
{
    public class StudentController : Controller
    {
        public List<Student> LoadStudents()
        {
            List<Student> students = new List<Student>();
            if (Directory.Exists(Config.GetStudentsPath()))
            {
                foreach (var studentFile in Directory.GetFiles(Config.GetStudentsPath()))
                {
                    if (new FileInfo(studentFile).Extension == ".rdf")
                    {
                        Student student = new Student();
                        XmlReader xmlReader = XmlReader.Create(studentFile);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                if (xmlReader.GetAttribute("rdf:about") != null)
                                {
                                    string[] studentDescription = xmlReader.GetAttribute("rdf:about")!.Split("#");
                                    if (studentDescription[1] != null)
                                    {
                                        student.StudentIdentifier = studentDescription[1];
                                    }
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
        public Student LoadStudent(string studentIdentifier)
        {
            Student student = new Student();

            if (Directory.Exists(Config.GetStudentsPath()))
            {
                foreach (var studentFile in Directory.GetFiles(Config.GetStudentsPath()))
                {
                    if (new FileInfo(studentFile).Extension == ".rdf")
                    {
                        XmlReader xmlReader = XmlReader.Create(studentFile);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {

                                if (xmlReader.GetAttribute("rdf:about") != null)
                                {
                                    string[] studentDescription = xmlReader.GetAttribute("rdf:about")!.Split("#");
                                    if (studentDescription[1] != null)
                                    {
                                        if(studentIdentifier == studentDescription[1])
                                        {
                                            student.StudentIdentifier = studentIdentifier;
                                        }
                                        else
                                        {
                                            break;
                                        }
                                    }
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
