using Microsoft.EntityFrameworkCore;
using DomainModel;
using Common;

namespace DataLayer
{
    public class CourseContext : DbContext
    {
        public CourseContext(DbContextOptions<CourseContext> options) : base(options)
        {
        }

        public DbSet<TestTemplate> TestTemplates { get; set; } = default!;
        public DbSet<QuestionTemplate> QuestionTemplates { get; set; } = default!;
        public DbSet<SubquestionTemplate> SubquestionTemplates { get; set; } = default!;
        public DbSet<User> Users { get; set; } = default!;
        public DbSet<Student> Students { get; set; } = default!;
        public DbSet<TestResult> TestResults { get; set; } = default!;
        public DbSet<QuestionResult> QuestionResults { get; set; } = default!;
        public DbSet<SubquestionResult> SubquestionResults { get; set; } = default!;
        public DbSet<UserRegistration> UserRegistrations { get; set; } = default!;
        public DbSet<GlobalSettings> GlobalSettings { get; set; } = default!;
        public DbSet<SubquestionTemplateRecord> SubquestionTemplateRecords { get; set; } = default!;
        public DbSet<SubquestionTemplateStatistics> SubquestionTemplateStatistics { get; set; } = default!;
        public DbSet<SubquestionResultRecord> SubquestionResultRecords { get; set; } = default!;
        public DbSet<SubquestionResultStatistics> SubquestionResultStatistics { get; set; } = default!;
        public DbSet<TestDifficultyStatistics> TestDifficultyStatistics { get; set; } = default!;
        public DbSet<Subject> Subjects { get; set; } = default!;

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<List<string>>().HasNoKey();

            modelBuilder.Entity<TestTemplate>().ToTable("TestTemplate");
            modelBuilder.Entity<TestTemplate>()
                .HasOne(t => t.Owner)
                .WithMany()
                .HasForeignKey(t => t.OwnerLogin)
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<TestTemplate>()
                .HasOne(t => t.Subject)
                .WithMany()
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<TestTemplate>().HasKey(t => new { t.TestTemplateId });

            modelBuilder.Entity<QuestionTemplate>().ToTable("QuestionTemplate");
            modelBuilder.Entity<QuestionTemplate>().HasKey(q => new { q.QuestionTemplateId });

            modelBuilder.Entity<SubquestionTemplate>().ToTable("SubquestionTemplate");
            modelBuilder.Entity<SubquestionTemplate>().HasKey(s => new { s.SubquestionTemplateId });
            modelBuilder.Entity<SubquestionTemplate>()
                .Property(e => e.CorrectAnswers)
                .HasConversion(
                v => string.Join(';', v),
                v => v.Split(';', StringSplitOptions.RemoveEmptyEntries));
            modelBuilder.Entity<SubquestionTemplate>()
                .Property(e => e.PossibleAnswers)
                .HasConversion(
                v => string.Join(';', v),
                v => v.Split(';', StringSplitOptions.RemoveEmptyEntries));
            modelBuilder.Entity<SubquestionTemplate>()
                .HasOne(s => s.QuestionTemplate)
                .WithMany()
                .HasForeignKey(s => new { s.QuestionTemplateId })
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<User>().ToTable("User");

            modelBuilder.Entity<Student>().ToTable("Student");
            modelBuilder.Entity<Student>()
                .HasMany(s => s.Subjects)
                .WithMany(st => st.Students);

            modelBuilder.Entity<TestResult>().ToTable("TestResult");
            modelBuilder.Entity<TestResult>()
                .HasOne(t => t.TestTemplate)
                .WithMany()
                .HasForeignKey(t => new { t.TestTemplateId })
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<TestResult>()
                .HasOne(t => t.Student)
                .WithMany()
                .HasForeignKey(t => new { t.StudentLogin })
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<TestResult>().HasKey(t => new { t.TestResultId });

            modelBuilder.Entity<QuestionResult>().ToTable("QuestionResult");
            modelBuilder.Entity<QuestionResult>()
                .HasOne(q => q.TestResult)
                .WithMany()
                .HasForeignKey(q => new { q.TestResultId })
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<QuestionResult>()
                .HasOne(q => q.QuestionTemplate)
                .WithMany()
                .HasForeignKey(q => new { q.QuestionTemplateId })
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<QuestionResult>().HasKey(q => new { q.QuestionResultId });

            modelBuilder.Entity<SubquestionResult>().ToTable("SubquestionResult");
            modelBuilder.Entity<SubquestionResult>()
                .HasOne(q => q.QuestionResult)
                .WithMany()
                .HasForeignKey(q => new { q.QuestionResultId })
                .OnDelete(DeleteBehavior.Cascade);
            modelBuilder.Entity<SubquestionResult>()
                .HasOne(q => q.SubquestionTemplate)
                .WithMany()
                .HasForeignKey(q => new { q.SubquestionTemplateId})
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<SubquestionResult>().HasKey(s => new { s.SubquestionResultId });
            modelBuilder.Entity<SubquestionResult>()
                .Property(e => e.StudentsAnswers)
                .HasConversion(
                v => string.Join(';', v),
                v => v.Split(';', StringSplitOptions.RemoveEmptyEntries));

            modelBuilder.Entity<UserRegistration>().ToTable("UserRegistration");
            
            modelBuilder.Entity<GlobalSettings>().ToTable("GlobalSettings");
            modelBuilder.Entity<GlobalSettings>().HasData(
                new GlobalSettings
                {
                    Id = 1,
                    TestingMode = false
                }
            );

            modelBuilder.Entity<SubquestionTemplateRecord>().ToTable("SubquestionTemplateRecord");
            modelBuilder.Entity<SubquestionTemplateRecord>().HasKey(s => new { s.SubquestionTemplateRecordId });
            modelBuilder.Entity<SubquestionTemplateRecord>()
                .HasOne(s => s.Owner)
                .WithMany()
                .HasForeignKey(s => s.OwnerLogin)
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<SubquestionTemplateRecord>()
                .HasOne(s => s.SubquestionTemplate)
                .WithMany()
                .HasForeignKey(s => new { s.SubquestionTemplateId })
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<SubquestionTemplateStatistics>().ToTable("SubquestionTemplateStatistics");
            modelBuilder.Entity<SubquestionTemplateStatistics>()
                .HasOne(s => s.User)
                .WithMany()
                .HasForeignKey(s => s.UserLogin)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<SubquestionResultRecord>().ToTable("SubquestionResultRecord");
            modelBuilder.Entity<SubquestionResultRecord>().HasKey(s => new { s.SubquestionResultRecordId });
            modelBuilder.Entity<SubquestionResultRecord>()
                .HasOne(s => s.Owner)
                .WithMany()
                .HasForeignKey(s => s.OwnerLogin)
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<SubquestionResultRecord>()
                .HasOne(s => s.SubquestionResult)
                .WithMany()
                .HasForeignKey(s => new { s.SubquestionResultId })
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<SubquestionResultStatistics>().ToTable("SubquestionResultStatistics");
            modelBuilder.Entity<SubquestionResultStatistics>()
                .HasOne(s => s.User)
                .WithMany()
                .HasForeignKey(s => s.UserLogin)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<TestDifficultyStatistics>().ToTable("TestDifficultyStatistics");
            modelBuilder.Entity<TestDifficultyStatistics>()
                .HasOne(s => s.User)
                .WithMany()
                .HasForeignKey(s => s.UserLogin)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<Subject>().ToTable("Subject");
            modelBuilder.Entity<Subject>().HasKey(s => new { s.SubjectId });
            modelBuilder.Entity<Subject>()
                .HasOne(s => s.Guarantor)
                .WithMany()
                .HasForeignKey(s => s.GuarantorLogin)
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<Subject>()
                .HasMany(s => s.Students)
                .WithMany(st => st.Subjects);
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            //this option should only be enabled during development phase
            optionsBuilder.EnableSensitiveDataLogging();
        }
    }
}
