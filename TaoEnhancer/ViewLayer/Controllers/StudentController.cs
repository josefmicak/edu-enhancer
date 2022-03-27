﻿using Common;
using Microsoft.AspNetCore.Mvc;
using System.Net.Mail;
using System.Xml;

namespace ViewLayer.Controllers
{
    public class StudentController : Controller
    {
        public List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)> LoadStudents()
        {
            List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)> students = new List<(string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email)>();
            
            if (Directory.Exists(Settings.GetStudentsPath()))
            {
                foreach (var studentFile in Directory.GetFiles(Settings.GetStudentsPath()))
                {
                    if (new FileInfo(studentFile).Extension == ".rdf")
                    {
                        (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) student = ("", "", "", "", "", "");
                        student.studentNumberIdentifier = Path.GetFileNameWithoutExtension(new FileInfo(studentFile).Name);

                        XmlReader xmlReader = XmlReader.Create(studentFile);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.studentIdentifier = xmlReader.GetAttribute("rdf:about").Split("#")[1];
                            }

                            if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.login = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userFirstName" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.firstName = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userLastName" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.lastName = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userMail" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.email = xmlReader.ReadInnerXml();
                            }
                        }

                        students.Add(student);
                    }
                }
            }
            else { throw Exceptions.StudentsPathNotFoundException; }

            return students;
        }

        public (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) LoadStudentByNumberIdentifier(string studentNumberIdentifier)
        {
            (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) student = (studentNumberIdentifier, "", "", "", "", "");

            if (Directory.Exists(Settings.GetStudentsPath()))
            {
                if (System.IO.File.Exists(Settings.GetStudentFilePath(studentNumberIdentifier)))
                {
                    XmlReader xmlReader = XmlReader.Create(Settings.GetStudentFilePath(studentNumberIdentifier));
                    while (xmlReader.Read())
                    {
                        if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            student.studentIdentifier = xmlReader.GetAttribute("rdf:about").Split("#")[1];
                        }

                        if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            student.login = xmlReader.ReadInnerXml();
                        }

                        if (xmlReader.Name == "ns0:userFirstName" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            student.firstName = xmlReader.ReadInnerXml();
                        }

                        if (xmlReader.Name == "ns0:userLastName" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            student.lastName = xmlReader.ReadInnerXml();
                        }

                        if (xmlReader.Name == "ns0:userMail" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            student.email = xmlReader.ReadInnerXml();
                        }
                    }
                }
                else { throw Exceptions.StudentFilePathNotFoundException; }
            }
            else { throw Exceptions.StudentsPathNotFoundException; }

            return student;
        }

        public (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) LoadStudentByIdentifier(string studentIdentifier)
        {
            (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) student = ("", studentIdentifier, "", "", "", "");

            if (Directory.Exists(Settings.GetStudentsPath()))
            {
                bool studentFound = false;

                foreach (var studentFile in Directory.GetFiles(Settings.GetStudentsPath()))
                {
                    if (new FileInfo(studentFile).Extension == ".rdf" && !studentFound)
                    {
                        student.studentNumberIdentifier = Path.GetFileNameWithoutExtension(new FileInfo(studentFile).Name);

                        XmlReader xmlReader = XmlReader.Create(studentFile);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "rdf:Description" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                string checkStudentIdentifier = xmlReader.GetAttribute("rdf:about").Split("#")[1];
                                if (studentIdentifier == checkStudentIdentifier)
                                {
                                    studentFound = true;
                                }
                                else
                                {
                                    studentFound = false;
                                    break;
                                }
                            }

                            if (xmlReader.Name == "ns0:login" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.login = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userFirstName" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.firstName = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userLastName" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.lastName = xmlReader.ReadInnerXml();
                            }

                            if (xmlReader.Name == "ns0:userMail" && xmlReader.NodeType != XmlNodeType.EndElement)
                            {
                                student.email = xmlReader.ReadInnerXml();
                            }
                        }
                    }
                }

                if (!studentFound) { throw Exceptions.StudentFilePathNotFoundException; }
            }
            else { throw Exceptions.StudentsPathNotFoundException; }

            return student;
        }

        public (string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email) LoadStudentByEmail(string loginEmail)
        {
            (string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email) student = (loginEmail, "", -1, "", "", "", "", "");

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
                                student.studentNumberIdentifier = splitImportedFileLineBySemicolon[0];
                                student.role = 0;

                                try
                                {
                                    (string studentNumberIdentifier, string studentIdentifier, string login, string firstName, string lastName, string email) loadedStudent = LoadStudentByNumberIdentifier(splitImportedFileLineBySemicolon[0]);
                                    student.studentIdentifier = loadedStudent.studentIdentifier;
                                    student.login = loadedStudent.login;
                                    student.firstName = loadedStudent.firstName;
                                    student.lastName = loadedStudent.lastName;
                                    student.email = loadedStudent.email;
                                }
                                catch(Exception e) { }

                                try
                                {
                                    student.role = int.Parse(splitImportedFileLineBySemicolon[1]);
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

            return student;
        }
    }
}