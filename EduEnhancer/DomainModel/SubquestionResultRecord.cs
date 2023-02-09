using CsvHelper.Configuration.Attributes;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class SubquestionResultRecord
    {
        [Ignore]
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int SubquestionResultRecordId { get; set; }
        [Ignore]
        public SubquestionResult SubquestionResult { get; set; } = default!;
        [Ignore]
        public User Owner { get; set; } = default!;
        [Ignore]
        public int SubquestionResultId { get; set; }
        [Ignore]
        public string OwnerLogin { get; set; } = default!;
        public double SubquestionTypeAveragePoints { get; set; }
        public double AnswerCorrectness { get; set; }
        public double SubjectAveragePoints { get; set; }
        public double WrongChoicePointsShare { get; set; }
        public int NegativePoints { get; set; }
        public double MinimumPointsShare { get; set; }
        public double StudentsPoints { get; set; }
    }
}
