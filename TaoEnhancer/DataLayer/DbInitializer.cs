using System;
using System.Linq;
using DomainModel;

namespace DataLayer
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

            if (context.TestResults.Any())
            {
                return;   // DB has been seeded
            }

            if (context.QuestionResults.Any())
            {
                return;   // DB has been seeded
            }

            if (context.SubquestionResults.Any())
            {
                return;   // DB has been seeded
            }

            if (context.Users.Any())
            {
                return;   // DB has been seeded
            }

            if (context.Students.Any())
            {
                return;   // DB has been seeded
            }

            if (context.UserRegistrations.Any())
            {
                return;   // DB has been seeded
            }

            if (context.GlobalSettings.Any())
            {
                return;   // DB has been seeded
            }

            if (context.SubquestionTemplateRecords.Any())
            {
                return;   // DB has been seeded
            }

            context.SaveChanges();
        }
    }
}
