using Common;
using Microsoft.AspNetCore.Mvc;
using System.Net.Mail;
using System.Xml;

namespace ViewLayer.Controllers
{
    public class StudentController : Controller
    {
        public (string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email) LoadStudentData(string loginEmail)
        {
            (string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email) studentData = (loginEmail, "", -1, "", "", "", "", "");

            if (Directory.Exists(Settings.GetStudentsPath()))
            {
                if (System.IO.File.Exists(Settings.GetStudentLoginDataPath(loginEmail)))
                {
                    string[] importedFileLines = System.IO.File.ReadAllLines(Settings.GetStudentLoginDataPath(loginEmail));
                    if(importedFileLines.Length > 0)
                    {
                        foreach (string importedFileLine in importedFileLines)
                        {
                            if (importedFileLine.Contains(';'))
                            {
                                string[] splitImportedFileLineBySemicolon = importedFileLine.Split(";");
                                studentData.studentNumberIdentifier = splitImportedFileLineBySemicolon[0];
                                studentData.role = 0;

                                try
                                {
                                    (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) studentInfo = LoadStudentInfo(splitImportedFileLineBySemicolon[0]);
                                    studentData.studentIdentifier = studentInfo.studentIdentifier;
                                    studentData.login = studentInfo.login;
                                    studentData.firstName = studentInfo.firstName;
                                    studentData.lastName = studentInfo.lastName;
                                    studentData.email = studentInfo.email;
                                }
                                catch(Exception e) { }

                                try
                                {
                                    studentData.role = int.Parse(splitImportedFileLineBySemicolon[1]);
                                }
                                catch (Exception e)
                                {
                                    break;
                                }
                            }
                            else { throw Exceptions.DataIdentifierNotFound; }
                        }
                    }
                    else { throw Exceptions.DataIdentifierNotFound; }
                }
                else { throw Exceptions.StudentLoginDataPathNotFoundException; }
            }
            else { throw Exceptions.StudentsPathNotFoundException; }

            return studentData;
        }

        public (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) LoadStudentInfo(string studentNumberIdentifier)
        {
            (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) studentInfo = (studentNumberIdentifier, "", "", "", "", "");

            if (Directory.Exists(Settings.GetStudentsPath()))
            {
                if (System.IO.File.Exists(Settings.GetStudentFilePath(studentNumberIdentifier)))
                {
                    XmlReader xmlReader = XmlReader.Create(Settings.GetStudentFilePath(studentNumberIdentifier));
                    while (xmlReader.Read())
                    {
                        if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            studentInfo.studentIdentifier = xmlReader.GetAttribute("rdf:about").Split("#")[1];
                        }

                        if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            studentInfo.login = xmlReader.ReadInnerXml();
                        }

                        if (xmlReader.Name == "ns0:userFirstName" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            studentInfo.firstName = xmlReader.ReadInnerXml();
                        }

                        if (xmlReader.Name == "ns0:userLastName" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            studentInfo.lastName = xmlReader.ReadInnerXml();
                        }

                        if (xmlReader.Name == "ns0:userMail" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            studentInfo.email = xmlReader.ReadInnerXml();
                        }
                    }
                }
                else { throw Exceptions.StudentFilePathNotFoundException; }
            }
            else { throw Exceptions.StudentsPathNotFoundException; }

            return studentInfo;
        }
    }
}
