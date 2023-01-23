using BusinessLayer;
using DataLayer;
using DomainModel;
using Common;
using Microsoft.AspNetCore.Hosting;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.EntityFrameworkCore.InMemory;
using ViewLayer.Controllers;
using Assert = NUnit.Framework.Assert;
using NUnit.Framework.Constraints;
using ArtificialIntelligenceTools;
using TestResult = DomainModel.TestResult;

namespace NUnitTests
{
    public class Tests
    {
        private readonly IConfiguration _configuration;

        public async Task<CourseContext> GetCourseContext()
        {
            
            var options = new DbContextOptionsBuilder<CourseContext>()
            .UseInMemoryDatabase(databaseName: "EduEnhancerDB")
            .Options;

            using (var _context = new CourseContext(options))
            {
                _context.Database.EnsureDeleted();

                //first test

                _context.Users.Add(new User
                {
                    Login = "nunittestingadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.Admin,
                    IsTestingData = true
                });
                _context.Students.Add(new Student
                {
                    Login = "nunittestingstudent",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    IsTestingData = true
                });
                _context.SaveChanges();

                _context.Subjects.Add(new Subject
                {
                    Abbreviation = "POS",
                    Name = "Počítačové sítě",
                    Guarantor = _context.Users.First(u => u.Login == "nunittestingadmin"),
                    GuarantorLogin = "nunittestingadmin",
                    IsTestingData = true,
                    Students = new List<Student> { _context.Students.First() }
                });
                _context.SaveChanges();

                //second test

                BusinessLayerFunctions businessLayerFunctions = new BusinessLayerFunctions(_context, _configuration);
                await businessLayerFunctions.AddAdmin("name", "surname", "login", "adminemail");
                await businessLayerFunctions.AddAdmin("Name", "Surname", "nunittestingteacher", "Email");
                await businessLayerFunctions.AddAdmin("Name", "Surname", "nunittestingmainadmin", "Email");
                _context.SaveChanges();

                DataFunctions dataFunctions = new DataFunctions(_context);
                TestTemplate testTemplateNUnit = new TestTemplate();
                testTemplateNUnit.Title = "NUnit title";
                testTemplateNUnit.MinimumPoints = 0;
                testTemplateNUnit.StartDate = new DateTime(2023, 12, 30, 00, 00, 00);
                testTemplateNUnit.EndDate = new DateTime(2023, 12, 31, 00, 00, 00);
                testTemplateNUnit.Subject = await _context.Subjects.FirstAsync(s => s.Abbreviation == "POS" && s.IsTestingData == true);
                testTemplateNUnit.Owner = await _context.Users.FirstAsync(u => u.Login == "login");
                testTemplateNUnit.OwnerLogin = "login";
                testTemplateNUnit.IsTestingData = true;
                testTemplateNUnit.QuestionTemplates = new List<QuestionTemplate>();
                await dataFunctions.AddTestTemplate(testTemplateNUnit);

                TestTemplate testTemplateNUnit2 = new TestTemplate();
                testTemplateNUnit2.Title = "NUnit title";
                testTemplateNUnit2.MinimumPoints = 0;
                testTemplateNUnit2.StartDate = new DateTime(2023, 12, 30, 00, 00, 00);
                testTemplateNUnit2.EndDate = new DateTime(2023, 12, 31, 00, 00, 00);
                testTemplateNUnit2.Subject = await _context.Subjects.FirstAsync(s => s.Abbreviation == "POS" && s.IsTestingData == true);
                testTemplateNUnit2.Owner = await _context.Users.FirstAsync(u => u.Login == "nunittestingteacher");
                testTemplateNUnit2.OwnerLogin = "nunittestingteacher";
                testTemplateNUnit2.IsTestingData = true;
                testTemplateNUnit2.QuestionTemplates = new List<QuestionTemplate>();
                await dataFunctions.AddTestTemplate(testTemplateNUnit2);

                QuestionTemplate questionTemplateNUnit = new QuestionTemplate();
                questionTemplateNUnit.Title = "NUnit title";
                questionTemplateNUnit.OwnerLogin = "login";
                questionTemplateNUnit.TestTemplate = await _context.TestTemplates.FirstAsync(t => t.Title == "NUnit title" && t.OwnerLogin == "login");
                questionTemplateNUnit.SubquestionTemplates = new List<SubquestionTemplate>();
                await dataFunctions.AddQuestionTemplate(questionTemplateNUnit);

                QuestionTemplate questionTemplateNUnit2 = new QuestionTemplate();
                questionTemplateNUnit2.Title = "NUnit title";
                questionTemplateNUnit2.OwnerLogin = "nunittestingteacher";
                questionTemplateNUnit2.TestTemplate = await _context.TestTemplates.FirstAsync(t => t.Title == "NUnit title" && t.OwnerLogin == "nunittestingteacher");
                questionTemplateNUnit2.SubquestionTemplates = new List<SubquestionTemplate>();
                await dataFunctions.AddQuestionTemplate(questionTemplateNUnit2);

                SubquestionTemplate subquestionTemplateNUnit2 = new SubquestionTemplate();
                subquestionTemplateNUnit2.SubquestionType = EnumTypes.SubquestionType.OrderingElements;
                subquestionTemplateNUnit2.SubquestionText = "";
                subquestionTemplateNUnit2.PossibleAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit2.CorrectAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit2.SubquestionPoints = 10;
                subquestionTemplateNUnit2.CorrectChoicePoints = 10 / 3;
                subquestionTemplateNUnit2.DefaultWrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit2.WrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit2.QuestionTemplate = await _context.QuestionTemplates.FirstAsync(q => q.OwnerLogin == "nunittestingteacher");
                subquestionTemplateNUnit2.QuestionTemplateId = subquestionTemplateNUnit2.QuestionTemplate.QuestionTemplateId;
                subquestionTemplateNUnit2.OwnerLogin = "nunittestingteacher";
                await dataFunctions.AddSubquestionTemplate(subquestionTemplateNUnit2, null, null);

                List<TestTemplate> testTemplates = DataGenerator.GenerateCorrelationalTestTemplates(new List<TestTemplate>(), 
                    100, _context.Subjects.Where(s => s.IsTestingData == true).ToList());
                List<SubquestionTemplate> subquestionTemplates = new List<SubquestionTemplate>();

                for (int i = 0; i < testTemplates.Count; i++)
                {
                    TestTemplate testTemplate = testTemplates[i];

                    for (int j = 0; j < testTemplate.QuestionTemplates.Count; j++)
                    {
                        QuestionTemplate questionTemplate = testTemplate.QuestionTemplates.ElementAt(j);

                        for (int k = 0; k < questionTemplate.SubquestionTemplates.Count; k++)
                        {
                            SubquestionTemplate subquestionTemplate = questionTemplate.SubquestionTemplates.ElementAt(k);
                            subquestionTemplate.QuestionTemplate = await _context.QuestionTemplates.FirstAsync(q => q.Title == "NUnit title");
                            subquestionTemplate.QuestionTemplateId = subquestionTemplate.QuestionTemplate.QuestionTemplateId;
                            subquestionTemplates.Add(subquestionTemplate);
                        }
                    }
                }

                for(int i = 0; i < subquestionTemplates.Count; i++)
                {
                    await dataFunctions.AddSubquestionTemplate(subquestionTemplates[i], null, null);
                }
                _context.SaveChanges();

                //third test

                TestTemplate testTemplateNUnit3 = new TestTemplate();
                testTemplateNUnit3.Title = "testTemplateNUnit3";
                testTemplateNUnit3.MinimumPoints = 0;
                testTemplateNUnit3.StartDate = new DateTime(2022, 12, 30, 00, 00, 00);
                testTemplateNUnit3.EndDate = new DateTime(2023, 12, 31, 00, 00, 00);
                testTemplateNUnit3.Subject = await _context.Subjects.FirstAsync(s => s.Abbreviation == "POS" && s.IsTestingData == true);
                testTemplateNUnit3.Owner = await _context.Users.FirstAsync(u => u.Login == "nunittestingmainadmin");
                testTemplateNUnit3.OwnerLogin = "nunittestingmainadmin";
                testTemplateNUnit3.IsTestingData = true;
                testTemplateNUnit3.QuestionTemplates = new List<QuestionTemplate>();
                await dataFunctions.AddTestTemplate(testTemplateNUnit3);

                QuestionTemplate questionTemplateNUnit3 = new QuestionTemplate();
                questionTemplateNUnit3.Title = "questionTemplateNUnit3";
                questionTemplateNUnit3.OwnerLogin = "nunittestingmainadmin";
                questionTemplateNUnit3.TestTemplate = await _context.TestTemplates.FirstAsync(t => t.Title == "testTemplateNUnit3" && t.OwnerLogin == "nunittestingmainadmin");
                questionTemplateNUnit3.SubquestionTemplates = new List<SubquestionTemplate>();
                await dataFunctions.AddQuestionTemplate(questionTemplateNUnit3);

                SubquestionTemplate subquestionTemplateNUnit31 = new SubquestionTemplate();
                subquestionTemplateNUnit31.SubquestionType = EnumTypes.SubquestionType.OrderingElements;
                subquestionTemplateNUnit31.SubquestionText = "";
                subquestionTemplateNUnit31.PossibleAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit31.CorrectAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit31.SubquestionPoints = 10;
                subquestionTemplateNUnit31.CorrectChoicePoints = 10 / 3;
                subquestionTemplateNUnit31.DefaultWrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit31.WrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit31.QuestionTemplate = await _context.QuestionTemplates.FirstAsync(q => q.OwnerLogin == "nunittestingmainadmin");
                subquestionTemplateNUnit31.QuestionTemplateId = subquestionTemplateNUnit31.QuestionTemplate.QuestionTemplateId;
                subquestionTemplateNUnit31.OwnerLogin = "nunittestingmainadmin";
                await dataFunctions.AddSubquestionTemplate(subquestionTemplateNUnit31, null, null);

                SubquestionTemplate subquestionTemplateNUnit32 = new SubquestionTemplate();
                subquestionTemplateNUnit32.SubquestionType = EnumTypes.SubquestionType.OrderingElements;
                subquestionTemplateNUnit32.SubquestionText = "";
                subquestionTemplateNUnit32.PossibleAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit32.CorrectAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit32.SubquestionPoints = 10;
                subquestionTemplateNUnit32.CorrectChoicePoints = 10 / 3;
                subquestionTemplateNUnit32.DefaultWrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit32.WrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit32.QuestionTemplate = await _context.QuestionTemplates.FirstAsync(q => q.OwnerLogin == "nunittestingmainadmin");
                subquestionTemplateNUnit32.QuestionTemplateId = subquestionTemplateNUnit32.QuestionTemplate.QuestionTemplateId;
                subquestionTemplateNUnit32.OwnerLogin = "nunittestingmainadmin";
                await dataFunctions.AddSubquestionTemplate(subquestionTemplateNUnit32, null, null);

                SubquestionTemplate subquestionTemplateNUnit33 = new SubquestionTemplate();
                subquestionTemplateNUnit33.SubquestionType = EnumTypes.SubquestionType.OrderingElements;
                subquestionTemplateNUnit33.SubquestionText = "";
                subquestionTemplateNUnit33.PossibleAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit33.CorrectAnswers = new string[] { "test1", "test2", "test3" };
                subquestionTemplateNUnit33.SubquestionPoints = 10;
                subquestionTemplateNUnit33.CorrectChoicePoints = 10 / 3;
                subquestionTemplateNUnit33.DefaultWrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit33.WrongChoicePoints = (10 / 3) * (-1);
                subquestionTemplateNUnit33.QuestionTemplate = await _context.QuestionTemplates.FirstAsync(q => q.OwnerLogin == "nunittestingmainadmin");
                subquestionTemplateNUnit33.QuestionTemplateId = subquestionTemplateNUnit33.QuestionTemplate.QuestionTemplateId;
                subquestionTemplateNUnit33.OwnerLogin = "nunittestingmainadmin";
                await dataFunctions.AddSubquestionTemplate(subquestionTemplateNUnit33, null, null);

                TestTemplate testTemplateNUnit4 = new TestTemplate();
                testTemplateNUnit4.Title = "testTemplateNUnit4";
                testTemplateNUnit4.MinimumPoints = 0;
                testTemplateNUnit4.StartDate = new DateTime(2022, 12, 30, 00, 00, 00);
                testTemplateNUnit4.EndDate = new DateTime(2023, 12, 31, 00, 00, 00);
                testTemplateNUnit4.Subject = await _context.Subjects.FirstAsync(s => s.Abbreviation == "POS" && s.IsTestingData == true);
                testTemplateNUnit4.Owner = await _context.Users.FirstAsync(u => u.Login == "nunittestingmainadmin");
                testTemplateNUnit4.OwnerLogin = "nunittestingmainadmin";
                testTemplateNUnit4.IsTestingData = true;
                testTemplateNUnit4.QuestionTemplates = new List<QuestionTemplate>();
                await dataFunctions.AddTestTemplate(testTemplateNUnit4);

                TestTemplate testTemplateNUnit5 = new TestTemplate();
                testTemplateNUnit5.Title = "testTemplateNUnit5";
                testTemplateNUnit5.MinimumPoints = 0;
                testTemplateNUnit5.StartDate = new DateTime(2022, 12, 30, 00, 00, 00);
                testTemplateNUnit5.EndDate = new DateTime(2023, 12, 31, 00, 00, 00);
                testTemplateNUnit5.Subject = await _context.Subjects.FirstAsync(s => s.Abbreviation == "POS" && s.IsTestingData == true);
                testTemplateNUnit5.Owner = await _context.Users.FirstAsync(u => u.Login == "nunittestingmainadmin");
                testTemplateNUnit5.OwnerLogin = "nunittestingmainadmin";
                testTemplateNUnit5.IsTestingData = true;
                testTemplateNUnit5.QuestionTemplates = new List<QuestionTemplate>();
                await dataFunctions.AddTestTemplate(testTemplateNUnit5);

                QuestionTemplate questionTemplateNUnit5 = new QuestionTemplate();
                questionTemplateNUnit5.Title = "questionTemplateNUnit5";
                questionTemplateNUnit5.OwnerLogin = "nunittestingmainadmin";
                questionTemplateNUnit5.TestTemplate = await _context.TestTemplates.FirstAsync(t => t.Title == "testTemplateNUnit5" && t.OwnerLogin == "nunittestingmainadmin");
                questionTemplateNUnit5.SubquestionTemplates = new List<SubquestionTemplate>();
                await dataFunctions.AddQuestionTemplate(questionTemplateNUnit5);
            }

            return new CourseContext(options);
        }

        [SetUp]
        public void Setup()
        {
        }
        
        private static readonly object[] _subjects =
        {
            //guarantor editing a subject - subject successfully edited
            new object[] {
                new Subject
                {
                    SubjectId = 1,
                    Abbreviation = "POS",
                    Name = "Počítačové sítě"
                },
                new User
                {
                    Login = "nunittestingadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.Admin,
                    IsTestingData = true
                },
                "Předmět byl úspěšně upraven."
            },

            //main editing a subject - subject successfully edited
            new object[] {
                new Subject
                {
                    SubjectId = 1,
                    Abbreviation = "POS",
                    Name = "Počítačové sítě"
                },
                new User
                {
                    Login = "nunittestingmainadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.MainAdmin,
                    IsTestingData = true
                },
                "Předmět byl úspěšně upraven."
            },

            //non-guarantor editing a subject - access denied
            new object[] {
                new Subject
                {
                    SubjectId = 1,
                    Abbreviation = "POS",
                    Name = "Počítačové sítě"
                },
                new User
                {
                    Login = "nunittestingteacher",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.Teacher,
                    IsTestingData = true
                },
                "Chyba: na tuto akci nemáte oprávnění"
            },

            //guarantor editing subject while omitting some necessary data - error message is sent back
            new object[] {
                new Subject
                {
                    SubjectId = 1,
                    Abbreviation = null,
                    Name = "Počítačové sítě"
                },
                new User
                {
                    Login = "nunittestingadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.Admin,
                    IsTestingData = true
                },
                "Chyba: U předmětu schází určité údaje."
            },

            //main admin editing subject while omitting some necessary data - error message is sent back
            new object[] {
                new Subject
                {
                    SubjectId = 1,
                    Abbreviation = null,
                    Name = "Počítačové sítě"
                },
                new User
                {
                    Login = "nunittestingmainadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.MainAdmin,
                    IsTestingData = true
                },
                "Chyba: U předmětu schází určité údaje."
            },

            //guarantor editing a subject with invalid id - exception is thrown
            new object[] {
                new Subject
                {
                    SubjectId = 2,
                    Abbreviation = "POS",
                    Name = "Počítačové sítě"
                },
                new User
                {
                    Login = "nunittestingadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.Admin,
                    IsTestingData = true
                },
                ""
            },

            //main admin editing a subject with invalid id - exception is thrown
            new object[] {
                new Subject
                {
                    SubjectId = 2,
                    Abbreviation = "POS",
                    Name = "Počítačové sítě"
                },
                new User
                {
                    Login = "nunittestingmainadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.MainAdmin,
                    IsTestingData = true
                },
                ""
            },

            //guarantor adds an invalid student to the "Students" property of Subject - error message is sent back
            new object[] {

                new Subject
                {
                    SubjectId = 1,
                    Abbreviation = "POS",
                    Name = "Počítačové sítě",
                    Students = new List<Student>() { new Student { Login = "newstudent" } }
                },
                new User
                {
                    Login = "nunittestingadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.Admin,
                    IsTestingData = true
                },
                "Při úpravě předmětu došlo k nečekané chybě."
            },

            //main admin adds an invalid student to the "Students" property of Subject - error message is sent back
            new object[] {

                new Subject
                {
                    SubjectId = 1,
                    Abbreviation = "POS",
                    Name = "Počítačové sítě",
                    Students = new List<Student>() { new Student { Login = "newstudent" } }
                },
                new User
                {
                    Login = "nunittestingmainadmin",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    Role = EnumTypes.Role.MainAdmin,
                    IsTestingData = true
                },
                "Při úpravě předmětu došlo k nečekané chybě."
            }
        };

       /* [Test]
        [TestCaseSource(nameof(_subjects))]
        public async Task EditSubjectTest(Subject subject, User user, string expectedResult)
        {
            CourseContext _context = await GetCourseContext();
            TemplateFunctions templateFunctions = new TemplateFunctions(_context);

            Subject? subjectNullable = await templateFunctions.GetSubjectByIdNullable(subject.SubjectId);
            if(subjectNullable == null)
            {
                Assert.That(async () => await templateFunctions.EditSubject(subject, user), Throws.Exception);
            }
            else
            {
                var result = await templateFunctions.EditSubject(subject, user);
                Assert.That(result, Is.EqualTo(expectedResult));
            }
        }*/

        private static readonly object[] _subquestionTemplates =
        {
            //admin with at least 100 added subquestion templates - gets a points suggestion for unsaved subquestion template
            new object[] {
                new SubquestionTemplate
                {
                    SubquestionType = EnumTypes.SubquestionType.OrderingElements,
                    PossibleAnswers = new string[] { "test1", "test2", "test3"},
                    CorrectAnswers = new string[] { "test1", "test2", "test3"},
                    OwnerLogin = "login"
                },
                false,
                "0"
            },

            //admin with at least 100 added subquestion templates - gets a points suggestion for saved subquestion template
            new object[] {
                new SubquestionTemplate
                {
                    SubquestionTemplateId = 10
                },
                true,
                "0"
            },

            //teacher with only 1 added subquestion template - message is sent back informing him that not enough subquestion templates have been added
            new object[] {
                new SubquestionTemplate
                {
                    SubquestionType = EnumTypes.SubquestionType.OrderingElements,
                    PossibleAnswers = new string[] { "test1", "test2", "test3"},
                    CorrectAnswers = new string[] { "test1", "test2", "test3"},
                    OwnerLogin = "nunittestingteacher"
                },
                false,
                "Pro použití této funkce je nutné přidat alespoň 100 zadání podotázek."
            },

            //teacher with only 1 added subquestion template - message is sent back informing him that not enough subquestion templates have been added
            new object[] {
                new SubquestionTemplate
                {
                    SubquestionTemplateId = 1
                },
                true,
                "Pro použití této funkce je nutné přidat alespoň 100 zadání podotázek."
            },

            //admin whose subquestion template statistics don't exist - exception is thrown
            new object[] {
                new SubquestionTemplate
                {
                    SubquestionType = EnumTypes.SubquestionType.OrderingElements,
                    PossibleAnswers = new string[] { "test1", "test2", "test3"},
                    CorrectAnswers = new string[] { "test1", "test2", "test3"},
                    OwnerLogin = "nunittestingadmin"
                },
                false,
                "Pro použití této funkce je nutné přidat alespoň 100 zadání podotázek."
            }
        };

       /* [Test]
        [TestCaseSource(nameof(_subquestionTemplates))]
        public async Task SubquestionTemplateSuggestionTest(SubquestionTemplate subquestionTemplate, bool subquestionTemplateExists, string expectedResult)
        {
            CourseContext _context = await GetCourseContext();
            TemplateFunctions templateFunctions = new TemplateFunctions(_context);
            subquestionTemplate.QuestionTemplate = await _context.QuestionTemplates.FirstAsync(q => q.QuestionTemplateId == 1);
            subquestionTemplate.QuestionTemplate.TestTemplate = await _context.TestTemplates.FirstAsync(t => t.TestTemplateId == 1);

            DataFunctions dataFunctions = new DataFunctions(_context);
            SubquestionTemplateStatistics? subquestionTemplateStatistics = await dataFunctions.GetSubquestionTemplateStatisticsDbSet().FirstOrDefaultAsync(s => s.UserLogin == subquestionTemplate.OwnerLogin);

            if (subquestionTemplateStatistics == null && !subquestionTemplateExists)
            {
                Assert.That(async () => await templateFunctions.GetSubquestionTemplatePointsSuggestion(subquestionTemplate, subquestionTemplateExists, true),
                    Throws.Exception);
            }
            else
            {
                var result = await templateFunctions.GetSubquestionTemplatePointsSuggestion(subquestionTemplate, subquestionTemplateExists, true);
                Assert.That(result, Is.EqualTo(expectedResult));
            }
        }*/

        private static readonly object[] _subquestionResults =
        {
            //non-existent student - exception is thrown
            new object[] {
                new SubquestionResult
                {
                    StudentsAnswers = new string[] { "a", "b", "c"}
                },
                1,
                new Student
                {
                    Login = "nonexistentttudent",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    IsTestingData = true
                },
                "testTemplateNUnit3"
            },

            //existing student with invalid subquestion result index - exception is thrown
            new object[] {
                new SubquestionResult
                {
                    StudentsAnswers = new string[] { "a", "b", "c"}
                },
                10,
                new Student
                {
                    Login = "nunittestingstudent",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    IsTestingData = true
                },
                "testTemplateNUnit3"
            },

            //existing student with valid subquestion result index - studentsAnswers field is modified
            new object[] {
                new SubquestionResult
                {
                    StudentsAnswers = new string[] { "a", "b", "c"}
                },
                1,
                new Student
                {
                    Login = "nunittestingstudent",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    IsTestingData = true
                },
                "testTemplateNUnit3"
            },

            //existing student filling out test with no questions - exception is thrown
            new object[] {
                new SubquestionResult
                {
                    StudentsAnswers = new string[] { "a", "b", "c"}
                },
                0,
                new Student
                {
                    Login = "nunittestingstudent",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    IsTestingData = true
                },
                "testTemplateNUnit4"
            },

            //existing student filling out test with one question and no subquestions - exception is thrown
            new object[] {
                new SubquestionResult
                {
                    StudentsAnswers = new string[] { "a", "b", "c"}
                },
                0,
                new Student
                {
                    Login = "nunittestingstudent",
                    Email = "Email",
                    FirstName = "Name",
                    LastName = "Surname",
                    IsTestingData = true
                },
                "testTemplateNUnit5"
            }
        };

       /* [Test]
        [TestCaseSource(nameof(_subquestionResults))]
        public async Task UpdateStudentsAnswersTest(SubquestionResult subquestionResult, int subquestionResultIndex, Student student, string testTemplateTitle)
        {
            CourseContext _context = await GetCourseContext();
            ResultFunctions resultFunctions = new ResultFunctions(_context);
            DataFunctions dataFunctions = new DataFunctions(_context);
            await resultFunctions.BeginStudentAttempt(await _context.TestTemplates
                .Include(t => t.QuestionTemplates)
                .ThenInclude(q => q.SubquestionTemplates)
                .FirstAsync(t => t.Title == testTemplateTitle),
            await _context.Students.FirstAsync(s => s.Login == "nunittestingstudent"));

            Student? studentCheck = await dataFunctions.GetStudentByLoginNullable(student.Login);
            QuestionResult? questionResult = await resultFunctions.GetQuestionResultDbSet()
                .Include(q => q.QuestionTemplate)
                .Include(q => q.QuestionTemplate.TestTemplate)
                .Include(q => q.SubquestionResults)
                .FirstOrDefaultAsync(q => q.OwnerLogin == "nunittestingmainadmin" && q.QuestionTemplate.TestTemplate.Title == testTemplateTitle);
            if(questionResult == null)
            {
                Assert.That(async () => await resultFunctions.UpdateSubquestionResultStudentsAnswers(subquestionResult, subquestionResultIndex, student),
                    Throws.Exception);
            }
            else if(subquestionResultIndex > questionResult.SubquestionResults.Count - 1)
            {
                Assert.That(async () => await resultFunctions.UpdateSubquestionResultStudentsAnswers(subquestionResult, subquestionResultIndex, student),
                    Throws.Exception);
            }
            else
            {
                if (studentCheck == null)
                {
                    SubquestionResult subquestionResult_ = questionResult.SubquestionResults.ElementAt(subquestionResultIndex);
                    Assert.That(async () => await resultFunctions.UpdateSubquestionResultStudentsAnswers(subquestionResult, subquestionResultIndex, student),
                        Throws.Exception);
                    Assert.That(subquestionResult_.StudentsAnswers, Is.Not.EqualTo(subquestionResult.StudentsAnswers));
                }
                else
                {
                    SubquestionResult subquestionResult_ = questionResult.SubquestionResults.ElementAt(subquestionResultIndex);
                    await resultFunctions.UpdateSubquestionResultStudentsAnswers(subquestionResult, subquestionResultIndex, student);
                    Assert.That(subquestionResult_.StudentsAnswers, Is.EqualTo(subquestionResult.StudentsAnswers));
                }
            }
        }*/
    }
}