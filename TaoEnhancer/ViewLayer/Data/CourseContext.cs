﻿using Microsoft.EntityFrameworkCore;
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
        public DbSet<Student> Students { get; set; }
        public DbSet<TestResult> TestResults { get; set; }
        public DbSet<QuestionResult> QuestionResults { get; set; }
        public DbSet<SubquestionResult> SubquestionResults { get; set; }

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
            modelBuilder.Entity<Student>().ToTable("Student");
            modelBuilder.Entity<TestResult>().ToTable("TestResult");
            modelBuilder.Entity<QuestionResult>().ToTable("QuestionResult");
            modelBuilder.Entity<QuestionResult>()
                .HasOne(q => q.TestResult)
                .WithMany()
                .HasForeignKey(q => q.TestResultIdentifier)
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<QuestionResult>().HasKey(q => new { q.TestResultIdentifier, q.QuestionNumberIdentifier });
            modelBuilder.Entity<SubquestionResult>().ToTable("SubquestionResult");
            modelBuilder.Entity<SubquestionTemplate>()
                .HasOne(s => s.QuestionTemplate)
                .WithMany()
                .HasForeignKey(s => s.QuestionNumberIdentifier)
                .OnDelete(DeleteBehavior.Restrict);
            modelBuilder.Entity<SubquestionResult>().HasKey(s => new { s.TestResultIdentifier, s.QuestionNumberIdentifier, s.SubquestionIdentifier });
            modelBuilder.Entity<SubquestionResult>()
                        .Property(e => e.StudentsAnswerList)
                        .HasConversion(
                        v => string.Join(',', v),
                        v => v.Split(',', StringSplitOptions.RemoveEmptyEntries));
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.EnableSensitiveDataLogging();
        }
    }
}
