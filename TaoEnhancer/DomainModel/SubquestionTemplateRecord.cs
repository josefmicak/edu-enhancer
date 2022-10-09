using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class SubquestionTemplateRecord
    {
        [Key]
        public string Id { get; set; } = default!;
        public double SubquestionTypeAveragePoints { get; set; }
        public double CorrectAnswersShare { get; set; }
        public double SubjectAveragePoints { get; set; }
        public int ContainsImage { get; set; }
        public int NegativePoints { get; set; }
        public double? MinimumPointsShare { get; set; }
        public double? SubquestionPoints { get; set; }
    }
}
