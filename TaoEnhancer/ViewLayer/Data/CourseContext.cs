using Microsoft.EntityFrameworkCore;
using DomainModel;

namespace ViewLayer.Data
{
    public class CourseContext : DbContext
    {
        public CourseContext(DbContextOptions<CourseContext> options) : base(options)
        {
        }

        public DbSet<TestTemplate> TestTemplates { get; set; }
        public DbSet<QuestionTemplate> QuestionTemplates { get; set; }
        public DbSet<SubquestionTemplate> SubquestionTemplates { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<List<string>>().HasNoKey();
            modelBuilder.Entity<SubquestionTemplate>()
                        .Property(e => e.CorrectAnswerList)
                        .HasConversion(
                        v => string.Join(',', v),
                        v => v.Split(',', StringSplitOptions.RemoveEmptyEntries));
            modelBuilder.Entity<SubquestionTemplate>()
                        .Property(e => e.PossibleAnswerList)
                        .HasConversion(
                        v => string.Join(',', v),
                        v => v.Split(',', StringSplitOptions.RemoveEmptyEntries));
            modelBuilder.Entity<TestTemplate>().ToTable("TestTemplate");
            modelBuilder.Entity<QuestionTemplate>().ToTable("QuestionTemplate");
            modelBuilder.Entity<SubquestionTemplate>().ToTable("SubquestionTemplate");
            modelBuilder.Entity<SubquestionTemplate>().HasKey(s => new { s.SubquestionIdentifier, s.QuestionNumberIdentifier });
        }
    }
}
