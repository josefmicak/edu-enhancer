using System.ComponentModel.DataAnnotations;

namespace DomainModel
{
    public class QuestionTemplate
    {
        public string QuestionNameIdentifier { get; set; }
        [Key]
        public string QuestionNumberIdentifier { get; set; }
        public string Title { get; set; }
        public string Label { get; set; }
        public TestTemplate TestTemplate { get; set; }
        public ICollection<SubquestionTemplate> SubquestionTemplateList { get; set; }
    }
}