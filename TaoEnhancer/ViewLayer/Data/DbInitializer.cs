using System;
using System.Linq;
using DomainModel;

namespace ViewLayer.Data
{
    public class DbInitializer
    {
        public static void Initialize(CourseContext context)
        {
            context.Database.EnsureCreated();

            // Look for any students.
            if (context.TestTemplates.Any())
            {
                return;   // DB has been seeded
            }

            if (context.QuestionTemplates.Any())
            {
                return;   // DB has been seeded
            }

            if (context.SubquestionTemplates.Any())
            {
                return;   // DB has been seeded
            }

            /*var testTemplates = new TestTemplate[]
            {
            new TestTemplate{TestNameIdentifier="One",TestNumberIdentifier="Two",Title="Three",QuestionTemplateList=null},
            new TestTemplate{TestNameIdentifier="Four",TestNumberIdentifier="Five",Title="Six",QuestionTemplateList=null}
            };
            foreach (TestTemplate testTemplate in testTemplates)
            {
                context.TestTemplates.Add(testTemplate);
            }*/
            context.SaveChanges();
        }
    }
}
