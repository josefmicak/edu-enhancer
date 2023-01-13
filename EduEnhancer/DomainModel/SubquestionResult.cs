using Common;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class SubquestionResult
    {
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int SubquestionResultId { get; set; }
        public int TestResultId { get; set; } = default!;
        public int QuestionTemplateId { get; set; } = default!;
        public int QuestionResultId { get; set; } = default!;
        public int SubquestionTemplateId { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public string[] StudentsAnswers { get; set; } = default!;
        public double StudentsPoints { get; set; } = default!;
        public double DefaultStudentsPoints { get; set; } = default!;
        public double AnswerCorrectness { get; set; }
        public EnumTypes.AnswerStatus AnswerStatus { get; set; }
        public SubquestionTemplate SubquestionTemplate { get; set; } = default!;
        public QuestionResult QuestionResult { get; set; } = default!;
    }
}