using Microsoft.EntityFrameworkCore;
using DomainModel;

namespace DataLayer
{
    public class CourseContext : DbContext
    {
        public CourseContext(DbContextOptions<CourseContext> options) : base(options)
        {
        }

        public DbSet<TestTemplate> TestTemplates { get; set; }
        public DbSet<QuestionTemplate> QuestionTemplates { get; set; }
        public DbSet<SubquestionTemplate> SubquestionTemplates { get; set; }
        public DbSet<User> Users { get; set; }
        public DbSet<Student> Students { get; set; }
        public DbSet<TestResult> TestResults { get; set; }
        public DbSet<QuestionResult> QuestionResults { get; set; }
        public DbSet<SubquestionResult> SubquestionResults { get; set; }
        public DbSet<UserRegistration> UserRegistrations { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<List<string>>().HasNoKey();

            modelBuilder.Entity<TestTemplate>().ToTable("TestTemplate");
            modelBuilder.Entity<TestTemplate>()
                .HasOne(q => q.Owner)
                .WithMany()
                .HasForeignKey(q => q.OwnerLogin)
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<TestTemplate>().HasKey(t => new { t.TestNumberIdentifier, t.OwnerLogin });

            modelBuilder.Entity<QuestionTemplate>().ToTable("QuestionTemplate");
            modelBuilder.Entity<QuestionTemplate>().HasKey(q => new { q.QuestionNumberIdentifier, q.OwnerLogin });

            modelBuilder.Entity<SubquestionTemplate>().ToTable("SubquestionTemplate");
            modelBuilder.Entity<SubquestionTemplate>().HasKey(s => new { s.SubquestionIdentifier, s.QuestionNumberIdentifier, s.OwnerLogin });
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
            modelBuilder.Entity<SubquestionTemplate>()
                .HasOne(s => s.QuestionTemplate)
                .WithMany()
                .HasForeignKey(s => new { s.QuestionNumberIdentifier, s.OwnerLogin })
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<User>().ToTable("User");

            modelBuilder.Entity<Student>().ToTable("Student");

            modelBuilder.Entity<TestResult>().ToTable("TestResult");
            modelBuilder.Entity<TestResult>()
                .HasOne(t => t.TestTemplate)
                .WithMany()
                .HasForeignKey(t => new { t.TestNumberIdentifier, t.OwnerLogin })
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<TestResult>()
                .HasOne(t => t.Student)
                .WithMany()
                .HasForeignKey(t => new { t.StudentLogin })
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<TestResult>().HasKey(t => new { t.TestResultIdentifier, t.OwnerLogin});

            modelBuilder.Entity<QuestionResult>().ToTable("QuestionResult");
            modelBuilder.Entity<QuestionResult>()
                .HasOne(q => q.TestResult)
                .WithMany()
                .HasForeignKey(q => new { q.TestResultIdentifier, q.OwnerLogin })
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<QuestionResult>()
                .HasOne(q => q.QuestionTemplate)
                .WithMany()
                .HasForeignKey(q => new { q.QuestionNumberIdentifier, q.OwnerLogin })
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<QuestionResult>().HasKey(q => new { q.TestResultIdentifier, q.QuestionNumberIdentifier, q.OwnerLogin });

            modelBuilder.Entity<SubquestionResult>().ToTable("SubquestionResult");
            modelBuilder.Entity<SubquestionResult>()
                .HasOne(q => q.QuestionResult)
                .WithMany()
                .HasForeignKey(q => new { q.TestResultIdentifier, q.QuestionNumberIdentifier, q.OwnerLogin })
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<SubquestionResult>()
                .HasOne(q => q.SubquestionTemplate)
                .WithMany()
                .HasForeignKey(q => new { q.SubquestionIdentifier, q.QuestionNumberIdentifier, q.OwnerLogin})
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<SubquestionResult>().HasKey(s => new { s.TestResultIdentifier, s.QuestionNumberIdentifier, s.SubquestionIdentifier, s.OwnerLogin });
            modelBuilder.Entity<SubquestionResult>()
                .Property(e => e.StudentsAnswerList)
                .HasConversion(
                v => string.Join(',', v),
                v => v.Split(',', StringSplitOptions.RemoveEmptyEntries));

            modelBuilder.Entity<UserRegistration>().ToTable("UserRegistration");
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.EnableSensitiveDataLogging();
        }
    }
}
