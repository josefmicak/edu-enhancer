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
                v => string.Join('~', v),
                v => v.Split('~', StringSplitOptions.RemoveEmptyEntries));
            modelBuilder.Entity<SubquestionTemplate>()
                .Property(e => e.PossibleAnswerList)
                .HasConversion(
                v => string.Join('~', v),
                v => v.Split('~', StringSplitOptions.RemoveEmptyEntries));
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
            modelBuilder.Entity<UserRegistration>()
                .HasOne(q => q.Student)
                .WithMany()
                .OnDelete(DeleteBehavior.SetNull);
            
            modelBuilder.Entity<GlobalSettings>().ToTable("GlobalSettings");
            modelBuilder.Entity<GlobalSettings>().HasData(
                new GlobalSettings
                {
                    Id = 1,
                    TestingMode = false
                }
            );

            modelBuilder.Entity<SubquestionTemplateRecord>().ToTable("SubquestionTemplateRecord");
            modelBuilder.Entity<SubquestionTemplateRecord>().HasKey(s => new { s.SubquestionIdentifier, s.QuestionNumberIdentifier, s.OwnerLogin });
            modelBuilder.Entity<SubquestionTemplateRecord>()
                .HasOne(s => s.Owner)
                .WithMany()
                .HasForeignKey(s => s.OwnerLogin)
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<SubquestionTemplateRecord>()
                .HasOne(s => s.SubquestionTemplate)
                .WithMany()
                .HasForeignKey(s => new { s.SubquestionIdentifier, s.QuestionNumberIdentifier, s.OwnerLogin })
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<SubquestionTemplateStatistics>().ToTable("SubquestionTemplateStatistics");
            modelBuilder.Entity<SubquestionTemplateStatistics>()
                .HasOne(s => s.User)
                .WithMany()
                .HasForeignKey(s => s.UserLogin)
                .OnDelete(DeleteBehavior.Cascade);

            modelBuilder.Entity<SubquestionResultRecord>().ToTable("SubquestionResultRecord");
            modelBuilder.Entity<SubquestionResultRecord>().HasKey(s => new { s.TestResultIdentifier, s.QuestionNumberIdentifier, s.SubquestionIdentifier, s.OwnerLogin });
            modelBuilder.Entity<SubquestionResultRecord>()
                .HasOne(s => s.Owner)
                .WithMany()
                .HasForeignKey(s => s.OwnerLogin)
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<SubquestionResultRecord>()
                .HasOne(s => s.SubquestionResult)
                .WithMany()
                .HasForeignKey(s => new { s.TestResultIdentifier, s.QuestionNumberIdentifier, s.SubquestionIdentifier, s.OwnerLogin })
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
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.EnableSensitiveDataLogging();
        }
    }
}
